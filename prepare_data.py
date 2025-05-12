"""
Data Preparation Script

This script loads the JSON dataset containing dialogue entries,
preprocesses the dialogues by concatenating the text turns, tokenizes
the data, and splits it into training and validation sets.

Usage:
    python prepare_data.py
"""

import json
import random
from typing import List, Dict

def load_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Load the dataset from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        List[Dict[str, str]]: List of dialogue dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_dialogues(data: List[Dict[str, str]]) -> List[str]:
    """
    Preprocess dialogues by concatenating dialogue turns into a single string.
    
    Each dialogue is represented as a dictionary with speaker names as keys.
    This function formats the dialogue as "Speaker: text" for each turn.
    
    Args:
        data (List[Dict[str, str]]): List of dialogues.
        
    Returns:
        List[str]: List of formatted dialogue strings.
    """
    formatted_dialogues = []
    for dialogue in data:
        dialogue_text = ""
        # Concatenate each dialogue turn with its speaker name.
        for speaker, text in dialogue.items():
            dialogue_text += f"{speaker}: {text} "
        formatted_dialogues.append(dialogue_text.strip())
    return formatted_dialogues

def split_data(data: List[str], train_ratio: float = 0.8) -> (List[str], List[str]):
    """
    Split the dialogues into training and validation sets.
    
    Args:
        data (List[str]): List of dialogues.
        train_ratio (float): Proportion of the data to use for training. Default is 0.8.
        
    Returns:
        (List[str], List[str]): A tuple containing training data and validation data.
    """
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

if __name__ == "__main__":
    file_path = "llm_train_text.json"
    # Load the JSON dataset.
    dataset = load_dataset(file_path)
    # Preprocess each dialogue.
    dialogues = preprocess_dialogues(dataset)
    # Split data into training and validation sets.
    train_data, val_data = split_data(dialogues)
    
    # Save the preprocessed dialogues to text files.
    with open("train_data.txt", "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(line + "\n")
    
    with open("val_data.txt", "w", encoding="utf-8") as f:
        for line in val_data:
            f.write(line + "\n")
    
    print(f"Data preparation completed. Training samples: {len(train_data)}, Validation samples: {len(val_data)}")