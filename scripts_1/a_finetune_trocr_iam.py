# 1_finetune_trocr_iam.py
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

# 1️⃣ Load the IAM handwriting dataset
dataset = load_dataset("Teklia/IAM-line")

# 2️⃣ Load pretrained TrOCR base model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# # For chinese
# processor = TrOCRProcessor.from_pretrained("lywen/trocr-chinese")
# model = VisionEncoderDecoderModel.from_pretrained("lywen/trocr-chinese")


# 🩵 FIX: define start token ID so decoder can begin generating
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# 3️⃣ Preprocess dataset: images → tensors, text → token IDs
def preprocess(batch):
    images = [x.convert("RGB") for x in batch["image"]]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    labels = processor.tokenizer(batch["text"],
                                 padding="max_length",
                                 max_length=64,
                                 truncation=True).input_ids
    batch["pixel_values"] = pixel_values
    batch["labels"] = labels
    return batch

dataset = dataset.map(preprocess, batched=True, remove_columns=["image", "text"])
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# 4️⃣ Define training configuration
training_args = Seq2SeqTrainingArguments(
    output_dir="../models/trocr_iam_finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    logging_steps=100,
    save_steps=300,
    num_train_epochs=1,   # you can increase later
    save_total_limit=2,
)

# 5️⃣ Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 6️⃣ Start fine-tuning
trainer.train()

# 7️⃣ Save model + processor
model.save_pretrained("../models/trocr_iam_finetuned")
processor.save_pretrained("../models/trocr_iam_finetuned")

# 🔧 Add this line to fix the error

print("✅ Fine-tuning complete! Model saved in ../models/trocr_iam_finetuned")
