"""
Check what labels the wambugu71 model actually has
"""
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME = "wambugu71/crop_leaf_diseases_vit"

print(f"Loading model: {MODEL_NAME}")
model = ViTForImageClassification.from_pretrained(MODEL_NAME)

print("\n=== Model Labels ===")
print(f"Number of labels: {model.num_labels}")
print("\nLabel mapping (id2label):")
for idx, label in model.config.id2label.items():
    print(f"  {idx}: {label}")

print("\n=== Label mapping (label2id) ===")
for label, idx in model.config.label2id.items():
    print(f"  {label}: {idx}")
