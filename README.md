# Dialogue-LLM-Finetuning

## Overview
This project demonstrates the complete process for building and training a large language model (LLM) on dialogue data. The process includes data preparation, model training, and evaluation. The key components include:

- **Data Preparation**:  
  - Loads the JSON dataset from `llm_train_text.json`.  
  - Preprocesses dialogues by concatenating individual dialogue turns with speaker labels.  
  - Splits the data into training (80%) and validation (20%) sets.  
  - Rationale: Formatting dialogues as a continuous text string with speaker labels helps the model understand conversational context. Tokenization and fixed-length sequences (with padding/truncation) ensure consistency during training.

- **Model Training**:  
  - Uses the pre-trained `distilgpt2` model and tokenizer from Hugging Face for fine-tuning.  
  - Implements a PyTorch training loop covering the forward pass, loss computation (using cross-entropy loss with ignored pad tokens), backpropagation, optimizer (AdamW) steps, learning rate scheduling, validation, and checkpointing.  
  - Rationale: `distilgpt2` is a compact model ideal for quick fine-tuning and demonstration. AdamW and a learning rate scheduler help in stabilizing and accelerating training.

- **Evaluation and Analysis**:  
  - The model's performance is monitored via validation loss.  
  - A sample dialogue is generated after training to qualitatively assess the model's conversational ability.

## Setup and Running Instructions

1. **Install Requirements**:  
   Ensure you have Python 3.7+ and install the necessary libraries:
   ```bash
   pip install torch transformers
   ```

2. **Prepare the Data**:  
   Place the dataset in a file named `llm_train_text.json` in the project root, then run:
   ```bash
   python prepare_data.py
   ```
   This will generate `train_data.txt` and `val_data.txt` for training and validation.

3. **Train the Model**:  
   Run the training script to fine-tune the model:
   ```bash
   python train.py
   ```
   Checkpoints will be saved after each epoch, and the validation loss will be printed.

4. **Evaluation**:
   Run the evaluation script to validate the model and generate outputs:
   ```bash
   python evaluate.py
   ```
