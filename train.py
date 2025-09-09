# # train_last_20_perc_layers 
# 
# So, in this notebook, we will freeze the first 80% of the layers (the embedding layer and the initial transformer blocks) since they likely capture basic features. Then, we will train the remaining 20% of the transformer layers along with the classification head.

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, Gemma3Model,  TrainingArguments, Trainer
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from utils import plot_and_save_losses, save_best_model, check_gpu_memory

# NOTE: we are using the pretrained model ( the model prior to SFT) since we have our own dataset
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL = "google/gemma-3-4b-pt"
SEED = 69

login(token=HUGGINGFACE_TOKEN)


#--------------------------------------------------
# LOAD DATA
#--------------------------------------------------
raw_dataset = load_dataset("mteb/tweet_sentiment_extraction")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

def tokenize_dataset(data):
    return tokenizer(data['text'], padding="max_length", truncation=True, max_length=128)

dataset = raw_dataset.map(tokenize_dataset, batched=True)

# shuffle the dataset and split into smaller part sow e can run on laptop
train = dataset['train'].shuffle(SEED).select(range(int(len(dataset['train']) * 0.7)))
dev = dataset['train'].shuffle(SEED).select(range(int(len(dataset['train']) * 0.7), len(dataset['train'])))

# make data into a tensor
X_train = torch.tensor(train['input_ids'])
y_train = F.one_hot(torch.tensor(train['label']), num_classes=3).float()
X_dev = torch.tensor(dev['input_ids'])
y_dev = F.one_hot(torch.tensor(dev['label']), num_classes=3).float()

train_dataset = TensorDataset(X_train, y_train)
dev_dataset = TensorDataset(X_dev, y_dev)
train_loader = DataLoader(train_dataset, batch_size=4)
dev_loader = DataLoader(dev_dataset, batch_size=4)

#--------------------------------------------------
# LOAD MODEL
#--------------------------------------------------


# Since we are using gemma we need to add on to the base model a classification head
# To do so we will import the base model then construct our model using output from the base model
baseModel = Gemma3Model.from_pretrained(MODEL, device_map='auto', 
                                        output_hidden_states=True, 
                                        attn_implementation="eager", 
                                        max_memory = {
                                        0: "20GiB",        # GPU 0 - more memory training
                                        1: "8GiB",        # GPU 1 - less of the model since it will have outpus and y 
                                        "cpu": "80Gib"
                                        }
                                        )

check_gpu_memory()


#this wont effect that mem taken up on the GPU but lets freze the firs 80% of layers and leave the reast to train
for param in baseModel.language_model.embed_tokens.parameters():
    param.requires_grad = False

max_layer_to_freeze = 26
for i, layer in enumerate(baseModel.language_model.layers):
    if i <= max_layer_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False


# We do this so that we have more room on the gpus
baseModel.vision_tower  = baseModel.vision_tower.to("cpu")
for param in baseModel.vision_tower.parameters():
                param.requires_grad = False
for param in baseModel.multi_modal_projector.parameters():
    param.requires_grad = False

check_gpu_memory()

# jsut set up some config stuff
baseModel.config.output_hidden_states = True   
baseModel.config.use_cache = False       
baseModel.gradient_checkpointing_enable()     

# we wrapp the base model with our models custom head
class Gemma3Classifier(nn.Module):
    def __init__(self, bmodel, hiddensize, dropout=0.1):
        super().__init__()
        self.bmodel = bmodel
        self.dropout = nn.Dropout(dropout) 
        self.head = nn.Linear(hiddensize, 3).to('cuda:1')
        self.device_placement = True
    
    def forward(self, input_ids):
        out = self.bmodel(input_ids)
        hidden_state = out.hidden_states[-1]
        embeddings = hidden_state[:, -1, :]  

        embeddings = embeddings.to('cuda:1')

        logits = self.head(self.dropout(embeddings))

        return logits 

model = Gemma3Classifier(bmodel=baseModel, dropout=0.1, hiddensize=baseModel.config.text_config.hidden_size)

# train loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters() ,lr=0.0003)
lossi = []
devlossi = []
torch.cuda.empty_cache()


accumulation_steps = 8 # 8 * 4(our small batch size due to mem constraints) = 32 new updates after
best_dev_loss = float('inf')
for epoch in tqdm(range(25)):
    model.train()

    loss_total = 0
    for i,  (X_train, y_train) in enumerate(train_loader):
        out = model(input_ids=X_train)
        y_train = y_train.to('cuda:1')
        loss = criterion(out, y_train)
        loss_total += loss.item() 

        loss = loss / accumulation_steps #since batch size is jsut 4
        loss.backward()

        # now we upadte every {accumulation_steps} 
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # if not perfectl;y diviable the we have left over gradient we need to use
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()        

    lossi.append(loss_total / len(train_loader))

    model.eval()
    dev_loss_total = 0
    with torch.no_grad():
        for X_dev, y_dev in dev_loader:
            out = model(input_ids=X_dev)
            y_dev = y_dev.to('cuda:1')
            loss = criterion(out, y_dev)
            dev_loss_total += loss.item()

    devlossi.append(dev_loss_total / len(dev_loader))

    plot_and_save_losses(lossi, devlossi)
    best_dev_loss = save_best_model(model, devlossi[-1], best_dev_loss, epoch)

    print(f"Epoch {epoch+1}: Train Loss: {lossi[-1]:.4f}, Dev Loss: {devlossi[-1]:.4f}")

