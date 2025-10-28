import argparse
import json
import sys
import os

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch


from torch import cuda, nn
from torch.utils.data import Dataset, DataLoader


from transformers import Trainer, TrainingArguments
from transformers import BertTokenizerFast, AutoTokenizer
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup


THRESHOLD = 0.5
focus_tags = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']

labels_to_ids = {j:i for i,j in enumerate(focus_tags)}
ids_to_labels = {i:j for i,j in enumerate(focus_tags)}

MAX_LEN = 128
LEARNING_RATE = 1e-05
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# Data retriever class, we need to tokenize text and binerize labels
# before feeding the Bert classifier, this class will help us handle
# this part on the fly

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):

        sentence = self.data.problem_description[index].strip()
        labels = self.data.tags[index]

        encoding = self.tokenizer(sentence,
                             return_offsets_mapping=True,
                             padding='max_length',
                             truncation=True,
                             max_length=self.max_len)

        labels_holder = np.array([0 for _ in range(len(labels_to_ids))], dtype=np.float32)
        for label in labels:
          labels_holder[labels_to_ids[label]] = 1
        labels = labels_holder[:]

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels)
        return item

    def __len__(self):
        return self.len





# We create an architecture based on a bert model for the embedding phase
# We add a dense layer for classification (we could have added a droupout layer in between)

class ModelClassifier(pl.LightningModule):

  def __init__(self, n_classes: int, model_name: str):
    super().__init__()
    self.bert = BertModel.from_pretrained(model_name, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.criterion = nn.BCELoss()

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss


  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(params=self.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=20,
      num_training_steps=100
    )
    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )



    


def parse_args():
    p = argparse.ArgumentParser(description="Predict codeforces problem labels")
    p.add_argument("--input-file", required=False, help="Path to saved problem json")
    p.add_argument("--test", required=False, help="Tests on a sample file at path code_classification_dataset/sample_100.json")
    return p.parse_args()

def main():
    args = parse_args()
    if args.test:
      args.input_file = '/content/code_classification_dataset/sample_100.json'
    if not args.input_file:
        raise SystemExit("Provide --input-file")

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

    model = ModelClassifier(len(labels_to_ids),'bert-base-uncased')
    state_dict = torch.load("model_v1",  weights_only=True, map_location=device)
    model.load_state_dict(state_dict)

    if args.input_file:
        with open(args.input_file, 'r') as f:
          data = json.load(f)


    data = pd.DataFrame([data])
    data.tags = data.tags.apply(lambda l: [t for t in l if t in focus_tags])
    data['problem_description'] = data['prob_desc_description'].fillna("") + " " + data['prob_desc_notes'].fillna("")  + " " + data['prob_desc_output_spec'].fillna("") + " " + data['prob_desc_input_spec'].fillna("")
    data_wrapper = dataset(data, tokenizer, MAX_LEN)

    data_params = {'batch_size': 1,
                'shuffle': True,
                'num_workers': 0
                }

    loader = DataLoader(data_wrapper, **data_params)


    with torch.no_grad():
      for idx, batch in enumerate(loader):

        ids = batch['input_ids']
        mask = batch['attention_mask']
        labels = batch['labels']

        loss, predictions = model(input_ids=ids, attention_mask=mask, labels=labels)
        predictions = (predictions > THRESHOLD).long()

    predictions = predictions.cpu().numpy().tolist()[0]
    predictions = [ids_to_labels[i] for i,pred in enumerate(predictions) if pred==1]
    return {"TRUE LABELS": data.tags.values.tolist()[0], "PREDICTED LABELS":predictions}



if __name__ == "__main__":
    result = main()
    print(result)