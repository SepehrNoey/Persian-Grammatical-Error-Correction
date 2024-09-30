# Persian Grammatical Error Correction using Transformer-Based Models

This project focuses on correcting grammatical and spelling errors in Persian sentences using two transformer-based models: **mT5** and **ParsBERT**. These are text-to-text models designed to handle Persian error correction tasks by transforming an erroneous sentence into a corrected one.

## Demo
Here we provide some generated outputs of the model. Notice that the model's performance is great and still needs several improvements, like the second example in the following table. As mentioned previously, the inputs and outputs of the model were in Persian language.

|Erroneous Input|Desired Output (Target)|Generated Output (Corrected)|
|--------|--------|--------|
|"بعنوان‌های" اعتباری مشهور شد|"به‌عنوان‌های" اعتباری مشهور شد|"به عنوان های" اعتباری مشهور شد|
|دریافت سفر هفته آینده سلطان قابوس به تهران یک سفر "پرتکلی" و معمولی نیست|دریافت سفر هفته آینده سلطان قابوس به تهران یک سفر "پروتکلی" و معمولی نیست|دریافت سفر هفته آینده سلطان قابوس به تهران یک سفر "عمیق" و معمولی نیست|

## Models

### Multilingual T5
The **mT5** model is trained on Persian grammatical error correction as a sequence-to-sequence (Seq2Seq) task. This model is the multilingual version of famous **T5 (Text-to-Text Transfer Transformer)** model which is capable of handling 101 languages. The architecture of T5 model can be seen in the following image.

<img src="https://github.com/user-attachments/assets/efb9c5b4-06b3-4e76-a777-8cfb1db1ecf6" height="400"/>

### ParsBERT (BERT-2-BERT)
ParsBERT is another transformer-based model that is specifically trained on Persian language. Since **ParsBERT** is an encoder-only model, we adapted it for the text-to-text task by employing two separate BERT models for the encoder and decoder, creating a **BERT-2-BERT** architecture. The architecture allows ParsBERT to handle grammar and spelling corrections similarly to the Seq2Seq model. The architecture of used model can be found in the following image.

<img src="https://github.com/user-attachments/assets/1986ab65-e9fd-4325-b3dd-e6d067a4bc9f" height="400"/>

## Dataset

We used the **PerSpellData** dataset for training. The dataset consists of approximately 6.4 million sentence pairs (sentences with errors and their corrected versions). These pairs are divided into two parts: **non-word errors** and **real-word errors**, both of which were utilized for this project.

The dataset is available at this [link](https://github.com/rominaoji/PerSpellData).

To optimize CPU and GPU RAM usage, the dataset pairs were tokenized and saved to disk. During training, the tokenized data was read one by one to reduce memory consumption.

## Monitoring and Metrics

During the training process, we monitored the model's performance using [**Weights and Biases (W&B)**](https://wandb.ai/site) for tracking losses and other metrics. The evaluation metrics used in this project included:

- **ROUGE Scores** (for sequence evaluation)
- **Word Error Rate (WER)** (for error detection)

The following image shows calculated scores for the model in the first 18000 steps:

|![evaluation_loss](https://github.com/user-attachments/assets/f9a72a01-4332-4b6d-a582-daf437f941d3)|![wer](https://github.com/user-attachments/assets/1a94e2f1-3750-4a9f-9756-6f7c2f2bf014)|
|----|----|
|![2_f](https://github.com/user-attachments/assets/c0867cbc-f2ef-451b-ad12-3baa92a819c7)|![l_f](https://github.com/user-attachments/assets/1617b068-884a-4a05-9f0e-eb1639a5e292)|


## Training Details

### mT5 Training Configuration

| **Parameter**                 | **Value**                            |
|-------------------------------|--------------------------------------|
| Programming Language           | Python                               |
| Frameworks                     | PyTorch, Huggingface Libraries, Pandas, Numpy |
| Model                          | mT5                                  |
| Checkpoint                     | [smartik/mt5-small-finetuned-gec-0.2](https://huggingface.co/smartik/mt5-small-finetuned-gec-0.2)  |
| Huggingface Model Class        | mT5                                  |
| Model Maximum Seq. Length      | 256                                  |
| Tokenizer                      | SentencePiece                        |
| Tokenizer Maximum Seq. Length  | 256                                  |
| Tuning Method                  | LoRA (Using `peft` library)          |
| Loss Function                  | CrossEntropyLoss                     |
| Data Collator                  | DataCollatorForSeq2Seq               |
| Initial Learning Rate          | 1e-3                                 |
| Epochs                         | 4                                    |
| Batch Size                     | 6 - 10                               |
| Learning Rate Scheduler        | Linear                               |
| LoRA Attention Dimension (r)   | 8                                    |
| LoRA Scaling Parameter (alpha) | 16                                   |
| LoRA Target Modules            | "q" and "v"                          |
| LoRA Dropout Probability       | 0.01                                 |
| Device Specs                   | Kaggle Virtual Machine – 12 GB RAM, GPU Tesla P100 16 GB RAM |

---

### ParsBERT Training Configuration

| **Parameter**                | **Value**                            |
|------------------------------|--------------------------------------|
| Programming Language          | Python                               |
| Frameworks                    | PyTorch, Huggingface Libraries, Pandas, Numpy |
| Model                         | ParsBERT                             |
| Checkpoint                    | [HooshvareLab/bert-fa-zwnj-base](https://huggingface.co/HooshvareLab/bert-fa-zwnj-base)        |
| Huggingface Model Class       | EncoderDecoderModel                  |
| Tokenizer                     | WordPiece                            |
| Tuning Method                 | Traditional Fine-Tuning              |
