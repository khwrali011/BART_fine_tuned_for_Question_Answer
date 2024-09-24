# BART Chatbot Fine-Tuning and Inference

This project demonstrates how to fine-tune the BART (Bidirectional and Auto-Regressive Transformers) model from Hugging Face's `transformers` library to create a chatbot using a custom dataset of question-answer pairs. The fine-tuned model can then be used for interactive chat with users.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Inference](#inference)
- [Model Saving](#model-saving)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository or download the files.
2. Install the necessary libraries:

   ```bash
   pip install transformers datasets torch
   ```

3. If using Google Colab, ensure that you have the necessary files uploaded, including the dataset in CSV format.

## Dataset

The dataset should be a CSV file with at least two columns: `question` and `answer`. This dataset will be used to fine-tune the BART model.

Example format of `Complete_Chatbot_Data.csv`:
```csv
question,answer
What is your name?,My name is BART.
How are you today?,I am a chatbot, I don't have feelings.
...
```

## Model Training

The training script fine-tunes the `facebook/bart-large` model using your dataset.

1. **Preprocessing the Data**:
   The `preprocess_data()` function converts the dataset into a format compatible with BART, where each row contains `input_text` (the question) and `target_text` (the answer).

2. **Tokenization**:
   The dataset is tokenized using BART's tokenizer to convert the text into tokens. Both the `input_text` and `target_text` are truncated and padded to fixed lengths for consistent training.

3. **Fine-Tuning**:
   Fine-tuning is handled using Hugging Face's `Trainer` API. The model is trained with the following default hyperparameters:
   - `learning_rate`: 2e-5
   - `num_train_epochs`: 3
   - `batch_size`: 4
   - `evaluation_strategy`: "epoch" (evaluates the model at the end of each epoch)

   To start training:
   ```python
   trainer.train()
   ```

4. **Device Handling**:
   The model will automatically use a GPU if available.

## Inference

Once the model is trained, you can use it for inference. The interactive chatbot allows users to input questions and receive model-generated responses.

To start a chat session, uncomment and run the `chat_with_bot` function in the script.

```python
def chat_with_bot():
    while True:
        user_input = input("You: ")
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
        output_ids = model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
        bot_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Bot: {bot_response}")

chat_with_bot()
```

## Model Saving

After training, the fine-tuned model and tokenizer are saved locally for later use or deployment. This includes both the model and tokenizer files:

```python
model.save_pretrained('./fine-tuned-bart')
tokenizer.save_pretrained('./fine-tuned-bart')
```

To compress the files for easier sharing:
```bash
!zip -r bart-large.zip /content/fine-tuned-bart
```

## Usage

1. Fine-tune the model using your dataset by following the steps in `Model Training`.
2. After training, use the `chat_with_bot()` function to interact with the chatbot.
3. Optionally, save the model and tokenizer to reuse later or deploy in different environments.

## Acknowledgements

- Hugging Face [Transformers](https://huggingface.co/transformers/) library for providing pre-trained models and the Trainer API.
- PyTorch for efficient model training.
- [Google Colab](https://colab.research.google.com/) for providing free access to GPUs.
