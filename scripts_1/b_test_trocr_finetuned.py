from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load your fine-tuned model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("../models/trocr_myhandwriting_1")  # your fine-tuned checkpoint
model.to(device)


# Load your handwritten image
image_path = "../test_photo/img_fine_tune_1/197.png"  # 👈 change this to your image path
image = Image.open(image_path).convert("RGB")

# Preprocess
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Generate text
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("📝 Predicted text:", generated_text)
