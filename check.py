print("SCRIPT STARTED")

from transformers import AutoModelForImageClassification

print("Transformers imported")

MODEL_NAME = "HurudzaAI/plantdiseasedetection1"

print("Loading model...")

model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")

print("Number of classes:", model.config.num_labels)

print("\nAll supported classes:")
print("-" * 50)

print(model.config.id2label)

for class_id, label in model.config.id2label.items():
    print(f"{class_id}: {label}")

print("\nSCRIPT FINISHED")