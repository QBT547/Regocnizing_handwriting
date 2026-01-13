import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
# import numpy as np
import os


# -------------------------------------------------------------
# LOAD ALL MODELS (only once)
# -------------------------------------------------------------
MODEL_PATHS = [
    "../models/trocr_iam_finetuned",
    # "../models/trocr_myhandwriting_1",
    # "../models/trocr_myhandwriting_2",
    # "../models/trocr_myhandwriting_Mukam_1_9",
    "../models/trocr_myhandwriting_all",
    "../models/trocr_myhandwriting_all_test",
]

models = []
processors = []

print("Loading models...")
for path in MODEL_PATHS:
    print(f"Loading: {path}")
    processor = TrOCRProcessor.from_pretrained(path)
    model = VisionEncoderDecoderModel.from_pretrained(path)
    model.eval()

    processors.append(processor)
    models.append(model)

print("All models loaded!")


# -------------------------------------------------------------
# GET CONFIDENCE (simple method)
# -------------------------------------------------------------
def calculate_confidence(logits):
    """
    Converts logit outputs into an average confidence score (0 to 1).
    """
    probs = torch.softmax(logits, dim=-1)
    max_probs = torch.max(probs, dim=-1).values
    return max_probs.mean().item()


# -------------------------------------------------------------
# RECOGNIZE FUNCTION (auto-select best model)
# -------------------------------------------------------------
def recognize_image(img_path):
    img = Image.open(img_path).convert("RGB")

    best_text = ""
    best_conf = -1
    best_model_index = -1

    print("\nEvaluating models...\n")

    for i, (processor, model) in enumerate(zip(processors, models)):

        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        output = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)

        text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]

        # Confidence from last step logits
        logits = output.scores[-1]
        conf = calculate_confidence(logits)

        print(f"[Model {i+1}] Text: {text}   | Confidence: {conf:.4f}")

        if conf > best_conf:
            best_conf = conf
            best_text = text
            best_model_index = i + 1

    print("\n--------------------------------")
    print(f"BEST MODEL: Model {best_model_index}")
    print(f"BEST CONFIDENCE: {best_conf:.4f}")
    print("--------------------------------\n")

    return best_text, best_model_index, best_conf


# -------------------------------------------------------------
# TEST
# -------------------------------------------------------------
if __name__ == "__main__":
    # result_list = []
    # for i in range(1,34):
    #     text, model_no, conf = recognize_image(f"../test_photo/check/{i}.jpg")
    #     # print(f"FINAL RESULT: {text}")
    #     result_list.append({i:text})

    # print(result_list)
    text, model_no, conf = recognize_image(f"../test_photo/check/1.jpg")
    print(f"FINAL RESULT: {text}")