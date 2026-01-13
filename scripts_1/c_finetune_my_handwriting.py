from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from PIL import Image
import torch
import os

# Paths
DATA_DIR = "../test_photo/"
MODEL_DIR = "../models/trocr_iam_finetuned"   # base fine-tuned model
# MODEL_DIR = "../models/trocr_myhandwriting_all/checkpoint-1500"
OUTPUT_DIR = "../models/trocr_myhandwriting_all_test"

# Load model + processor (IMPORTANT: load SAME processor used before)
processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)

# Proper configs
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id

# Load dataset
image_dir = os.path.join(DATA_DIR, "check")
label_file = os.path.join(DATA_DIR, "check.txt")

data = []
with open(label_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)

        if len(parts) < 2:
            print("Skipping invalid line:", line)
            continue

        img_name, text = parts
        data.append({
            "image": os.path.join(image_dir, img_name),
            "text": text
        })

dataset = Dataset.from_list(data)

# Preprocessing
def preprocess(batch):
    images = [Image.open(p).convert("RGB") for p in batch["image"]]
    pixel_values = processor(images=images, return_tensors="pt", padding=True).pixel_values

    encoding = processor.tokenizer(
        batch["text"],
        padding="max_length",
        max_length=64,
        truncation=True
    )

    labels = encoding.input_ids

    # Mask padding tokens for labels
    labels = [
        [(token if token != processor.tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]

    batch["pixel_values"] = pixel_values
    batch["labels"] = torch.tensor(labels)

    return batch

dataset = dataset.map(preprocess, batched=True, batch_size=4)

# Training settings
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=6,
    learning_rate=5e-5,
    save_total_limit=2,
    logging_steps=10,
    fp16=True,                     # If GPU available
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)   # ALWAYS SAVE PROCESSOR TOO

print(f"🔥 Fine-tuning complete! Model saved to {OUTPUT_DIR}")
