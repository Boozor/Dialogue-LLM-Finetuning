# Report

## Overview

This project demonstrates the process of building and training a Large Language Model (LLM) using a provided text dialogue dataset. The provided JSON dataset "llm_train_text.json" was used to fine-tune a pre-trained model ("distilgpt2") from the Hugging Face Transformers library. The project is divided into three main phases:
1. Data Preparation
2. Model Training
3. Evaluation and Analysis

## 1. Data Preparation

### Steps Taken:
- **Loading the Data:**  
  The JSON dataset is loaded from the file `llm_train_text.json`. Each JSON entry represents a dialogue between two speakers (e.g., Alex and Bob).  
- **Preprocessing:**  
  The dialogues are preprocessed by concatenating the turns from each speaker with their respective labels into a single string. This formatting (e.g., "Alex: Hello... Bob: I'm fine...") preserves the conversational context while making it suitable for language modeling.  
- **Tokenization, Padding, and Truncation:**  
  The preprocessed text is then tokenized using the `distilgpt2` tokenizer. Padding and truncation are applied to guarantee uniform sequence lengths, which simplifies batching during model training.  
- **Data Splitting:**  
  The dataset is split into training (80%) and validation (20%) sets. This split ensures that model performance can be monitored and validated against unseen data.

### Rationale:
- **Dialogue Formatting:**  
  By maintaining speaker labels, the model learns who is speaking and is better able to generate contextually appropriate responses.  
- **Uniform Sequence Lengths:**  
  Consistency in sequence lengths (achieved via padding/truncation) is vital for leveraging GPU acceleration and simplifying the model’s batch processing.
- **Train/Validation Split:**  
  The split allows for ongoing evaluation during training, helping detect overfitting and guiding hyperparameter tuning.

## 2. Model Selection and Training

### Model & Toolkit:
- **Model Choice: "distilgpt2"**  
  I selected "distilgpt2" as it is a compact variant of GPT2, making it more feasible to fine-tune on a small dataset. Its proven language generation capabilities make it a good candidate for dialogue generation.
- **Framework:**  
  PyTorch was used because of its flexible tensor operations and integration with Hugging Face Transformers.
- **Tokenizer:**  
  The tokenizer corresponding to "distilgpt2" ensures that the text format and special tokens (like EOS and PAD) are correctly handled.

### Training Process:
- **PyTorch Training Loop:**  
  A standard training loop is implemented, which includes:
  - **Forward Pass:** Processing input sequences to generate logits over the vocabulary.
  - **Loss Calculation:** Using cross-entropy loss with the pad token ignored (to avoid penalizing the model for padded values).
  - **Backpropagation & Optimizer Step:** Utilizing the AdamW optimizer, which is effective for transformer models due to its weight decay regularization.
  - **Learning Rate Scheduling:** A linear scheduler is applied to adjust the learning rate as training progresses.
  - **Checkpointing:** The model’s state is saved after each epoch. This is essential for resuming training and for evaluating the model at different stages.

### Rationale:
- **Model Efficiency:**  
  "distilgpt2" provides a balance between performance and computational requirements.
- **Optimization Strategy:**  
  AdamW and a learning rate scheduler together help in stabilizing the training process and in dealing with potential issues like exploding gradients.
- **Checkpointing:**  
  Checkpoints allow for iterative improvements and facilitate error analysis by evaluating model performance at different training stages.

## 3. Evaluation and Analysis

### Evaluation Components:
- **Validation Loss:**  
  The validation dataset is used to compute the loss after each training epoch. This quantitative metric helps in understanding if the model is generalizing well to new data.
- **Sample Generation:**  
  A sample dialogue is generated using a pre-defined prompt, enabling qualitative assessment of the model’s dialogue handling capabilities.

### Analysis of Model Strengths and Weaknesses:
- **Strengths:**
  - **Fluency:**  
    The model produces natural-sounding, grammatically correct text.
- **Weaknesses:**
  - **Overfitting:**  
    The validation loss lags behind the training loss, this indicates that the model is overfitting on a small dataset.
  - **Lack of Diversity:**  
    The generated dialogue is too generic. This is because the training dataset does not contain a diverse range of examples.
  - **Context Understanding:**  
    Sometimes, the model loses track of context, indicating areas for data augmentation.
  - **Incoherence and Irrelevance:**  
    The generated dialogue sometimes does not show a good understanding of context. This is a result of limited training data

### Rationale:
- **Combining Quantitative and Qualitative Analysis:**  
  Validation loss provides an objective measure, while sample generation offers insights into practical performance and user experience.
- **Iterative Improvement:**  
  Analyzing strengths and weaknesses allows for targeted improvements, such as better data augmentation or adjustments in model hyperparameters.
