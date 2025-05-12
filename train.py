"""
Model Training Script

This script fine-tunes the pre-trained `distilgpt2` model using the processed dialogue data.
It includes a PyTorch training loop with forward pass, loss computation, backpropagation,
optimizer steps, learning rate scheduling, and checkpointing.

Usage:
    python train.py
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
#from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW 

class DialogueDataset(Dataset):
    """
    PyTorch Dataset for dialogue data.

    Each example in the dataset is a tokenized sequence obtained from the dialogue text.
    """
    def __init__(self, file_path: str, tokenizer, block_size: int = 128):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for line in lines:
            # Tokenize with padding and truncation.
            tokenized_output = tokenizer.encode_plus(
                line,
                truncation=True,
                max_length=block_size,
                padding="max_length",
                return_tensors="pt"
            )
            self.examples.append(tokenized_output["input_ids"].squeeze())
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train(model, tokenizer, train_loader, epochs=3, learning_rate=5e-5, device="cpu"):
    """
    Training loop for fine-tuning the model.

    Args:
        model: Pre-trained language model.
        tokenizer: Tokenizer associated with the model.
        train_loader: DataLoader for training data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device for computation (e.g., 'cuda' or 'cpu').
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    
    # Learning rate scheduler to decrease the learning rate gradually.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch.to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}")
        
        # Save a checkpoint after each epoch.
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    # Hyperparameters and file paths.
    train_file = "train_data.txt"
    epochs = 3
    batch_size = 2
    learning_rate = 5e-5
    block_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the pre-trained tokenizer and model.
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    # Set pad token if undefined.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    
    # Create PyTorch dataset and dataloader.
    train_dataset = DialogueDataset(train_file, tokenizer, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Start training.
    train(model, tokenizer, train_loader, epochs, learning_rate, device)