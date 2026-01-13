from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from PIL import Image
import numpy as np
import torch
import os

# --- Paths (edit if necessary) ---
DATA_DIR = "../test_photo/"
BASE_PROCESSOR_DIR = "../models/trocr_iam_finetuned"   # where your processor / tokenizer lives
CHECKPOINT_DIR = "../models/trocr_myhandwriting_all/checkpoint-1500"  # latest checkpoint
OUTPUT_DIR = "../models/trocr_myhandwriting_all"       # final output dir (trainer will write checkpoints here)

# --- Load processor and model correctly ---
# processor must come from the original model/processor folder (not the checkpoint)
processor = TrOCRProcessor.from_pretrained(BASE_PROCESSOR_DIR)

# load model weights from the checkpoint
model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT_DIR)

# make sure special token ids are set
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id

# --- Build dataset from your label file ---
image_dir = os.path.join(DATA_DIR, "all")
label_file = os.path.join(DATA_DIR, "all_labels_.txt")

data = []
with open(label_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) < 2:
            print("Skipping invalid line:", line)
            continue
        img_name, text = parts
        data.append({"image": os.path.join(image_dir, img_name), "text": text})

dataset = Dataset.from_list(data)

# --- Preprocessing function ---
def preprocess(batch):
    # load images
    images = [Image.open(p).convert("RGB") for p in batch["image"]]

    # processor returns a dict; use numpy so dataset.map returns numpy arrays that can be converted to torch later
    enc_imgs = processor(images=images, return_tensors="np", padding=True)
    pixel_values = enc_imgs.pixel_values  # numpy array shape (batch, C, H, W)

    # tokenize labels
    encoding = processor.tokenizer(
        batch["text"],
        padding="max_length",
        max_length=64,
        truncation=True,
    )
    labels = encoding["input_ids"]  # list of lists (batch, seq_len)

    # mask padding tokens with -100 so loss ignores them
    pad_id = processor.tokenizer.pad_token_id
    labels = [[(tok if tok != pad_id else -100) for tok in label] for label in labels]

    return {"pixel_values": pixel_values, "labels": np.array(labels, dtype=np.int64)}

# Map and remove the original columns (image/text) so dataset only has inputs for Trainer
dataset = dataset.map(preprocess, batched=True, batch_size=4, remove_columns=["image", "text"])

# Convert dataset format to PyTorch tensors (Trainer expects tensors)
dataset.set_format(type="torch", columns=["pixel_values", "labels"])

# --- Training settings ---
use_fp16 = torch.cuda.is_available()  # enable fp16 only if CUDA available

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=6,
    learning_rate=5e-5,
    save_total_limit=2,
    logging_steps=10,
    fp16=use_fp16,
    # You can add other arguments you need, e.g. gradient_accumulation_steps, save_steps, etc.
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

# --- Resume training from the checkpoint you loaded above ---
# Using the checkpoint path ensures optimizer/scheduler states are restored
trainer.train(resume_from_checkpoint=CHECKPOINT_DIR)

# --- Save final model + processor ---
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"🔥 Fine-tuning complete! Model & processor saved to {OUTPUT_DIR}")
