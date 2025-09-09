import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import login
from transformers import Gemma3Model
from sklearn.metrics import f1_score
import numpy as np

from utils import check_gpu_memory

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

def load_model(model_path):
    """Load the trained model from checkpoint"""
    # Load environment variables
    load_dotenv()
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MODEL = "google/gemma-3-4b-pt"
    
    # Login to Hugging Face
    login(token=HUGGINGFACE_TOKEN)
    
    # Load base model with same configuration as training
    baseModel = Gemma3Model.from_pretrained(
        MODEL, 
        device_map='auto', 
        output_hidden_states=True, 
        attn_implementation="eager",
        max_memory={
            0: "20GiB",  # GPU 0 - more memory for training
            1: "8GiB",   # GPU 1 - less of the model since it will have outputs and y
            "cpu": "80GiB"
        }
    )
    
    check_gpu_memory()
    
    #creat model and load the weights
    hidden_size = baseModel.config.hidden_size
    model = Gemma3Classifier(baseModel, hidden_size)
    weights = torch.load(model_path, map_location='cpu')
    model.load_state_dict(weights)
    
    model.eval()
    return model

def load_data():
    """Load and preprocess the test dataset"""
    # Load environment variables
    load_dotenv()
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MODEL = "google/gemma-3-4b-pt"
    SEED = 69
    
    # Login to Hugging Face
    login(token=HUGGINGFACE_TOKEN)
    
    # Load dataset
    raw_dataset = load_dataset("mteb/tweet_sentiment_extraction")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    
    def tokenize_dataset(data):
        return tokenizer(data['text'], padding="max_length", truncation=True, max_length=128)
    
    dataset = raw_dataset.map(tokenize_dataset, batched=True)
    
    # Shuffle the dataset and split into smaller part so we can run on laptop
    test = dataset['test'].shuffle(SEED)
    
    # Make data into tensors
    X_test = torch.tensor(test['input_ids'])
    y_test = F.one_hot(torch.tensor(test['label']), num_classes=3).float()
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    return test_loader, dataset, tokenizer

def evaluate_model(model, test_loader):
    """Evaluate the model and calculate F1 score"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # Move inputs to appropriate device
            if torch.cuda.is_available():
                inputs = inputs.to('cuda:0')  # Base model is on cuda:0
            
            # Get predictions
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(labels, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
            
            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx * inputs.size(0)} samples...")
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    print(f"\nEvaluation Results:")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return f1, accuracy

def show_examples(model, dataset, tokenizer):
    """Show a few example classifications"""
    print("\n" + "="*50)
    print("EXAMPLE CLASSIFICATIONS")
    print("="*50)
    
    # Get a few examples
    for i in [4, 10, 25, 50, 100]:
        if i >= len(dataset['train']):
            continue
            
        ex = dataset['train'][i]
        ex_text = ex['text']
        ex_input = torch.tensor(ex['input_ids']).unsqueeze(dim=0)
        ex_label = ex['label']
        
        # Move to appropriate device
        if torch.cuda.is_available():
            ex_input = ex_input.to('cuda:0')
        
        with torch.no_grad():
            pred = model(ex_input)
            pred_probs = torch.softmax(pred, dim=1)
            pred_class = torch.argmax(pred, dim=1).item()
        
        print(f'\nExample {i+1}:')
        print(f'Text: {ex_text}')
        
        # Convert label to one-hot format for display
        if ex_label == 0:
            true_label_str = '[1, 0, 0] (Negative)'
        elif ex_label == 1:
            true_label_str = '[0, 1, 0] (Neutral)'
        else:
            true_label_str = '[0, 0, 1] (Positive)'
        
        pred_label_str = ['Negative', 'Neutral', 'Positive'][pred_class]
        
        print(f'True label: {true_label_str}')
        print(f'Prediction: {pred_probs.cpu().numpy().flatten()} -> {pred_label_str}')
        print(f'Correct: {"✓" if pred_class == ex_label else "✗"}')
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Gemma3 Sentiment Classifier')
    parser.add_argument('model_path', type=str, help='Path to the trained model checkpoint')
    args = parser.parse_args()
    
    print("Loading model...")
    model = load_model(args.model_path)
    
    print("Loading test data...")
    test_loader, dataset, tokenizer = load_data()
    
    print("Starting evaluation...")
    f1, accuracy = evaluate_model(model, test_loader)
    
    print(f"\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print("Showing example classifications...")
    show_examples(model, dataset, tokenizer)

if __name__ == "__main__":
    main()