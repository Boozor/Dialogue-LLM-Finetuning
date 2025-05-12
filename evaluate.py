"""
Model Evaluation and Sample Generation Script

This script loads a fine-tuned model checkpoint and evaluates its performance on the validation data.
It also generates a sample dialogue using a provided prompt.

Usage:
    python evaluate.py --checkpoint checkpoint_epoch_3.pt
    (Adjust the --checkpoint argument as needed.)
"""

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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


def validate(model, tokenizer, val_loader, device="cpu"):
    """
    Validate the fine-tuned model on the validation dataset.

    Args:
        model: The fine-tuned model.
        tokenizer: Tokenizer.
        val_loader: DataLoader for validation data.
        device (str): Computation device.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch.to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")


def generate_sample(model, tokenizer, prompt: str, max_length: int = 50, device="cpu") -> str:
    """
    Generate a sample dialogue from a given prompt.

    Args:
        model: The trained language model.
        tokenizer: Associated tokenizer.
        prompt (str): Prompt text to start generation.
        max_length (int): Maximum sequence length.
        device (str): Computation device.

    Returns:
        str: The generated dialogue text.
    """
    model.eval()
    # Tokenize the prompt with attention mask.
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device) #add special token here
    attention_mask = torch.ones_like(input_ids).to(device) #create attention_mask
    
    #print(input_ids)
    #print(attention_mask)
    output_ids = model.generate(
        input_ids=input_ids,  # Pass input_ids here
        attention_mask=attention_mask,  # Pass attention_mask here
        max_length=max_length,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id #pad_token_id here
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LLM model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--val_file", type=str, default="val_data.txt", help="Validation data file path.")
    parser.add_argument("--prompt", type=str, default="Alex: Hi, how are you? Bob:", help="Prompt for sample generation.")
    args = parser.parse_args()
    '''

    class Args:
      checkpoint = "checkpoint_epoch_3.pt" # checkpoint path
      val_file = "val_data.txt"
      prompt = "Alex: Hi, how are you? Bob:"
    args = Args()
    

    # Device configuration.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load tokenizer and model.
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    
    # Load the checkpoint.
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    print(f"Loaded model from checkpoint: {args.checkpoint}")
    
    # Create validation dataset and loader.
    block_size = 128
    batch_size = 2
    from torch.utils.data import DataLoader
    val_dataset = DialogueDataset(args.val_file, tokenizer, block_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Validate model performance.
    validate(model, tokenizer, val_loader, device)
    
# Generate and display a sample dialogue.
generated_dialogue = generate_sample(model, tokenizer, args.prompt, device=device)
print("Generated dialogue sample:")
print(generated_dialogue)