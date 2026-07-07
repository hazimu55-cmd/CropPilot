"""
Fine-tune ViT model on Indian crop disease dataset
"""
import os
import torch
from datasets import load_from_disk
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image
import evaluate

# Configuration
DATA_DIR = "data/processed_dataset"
MODEL_OUTPUT_DIR = "models/crop_disease_vit"
BASE_MODEL = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    """Load the prepared dataset"""
    print(f"Loading dataset from {DATA_DIR}...")
    dataset = load_from_disk(DATA_DIR)
    
    # Split into train/validation if not already split
    if "validation" not in dataset:
        split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        dataset = {
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        }
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    
    # Get label names
    label_names = dataset["train"].features["label"].names
    num_labels = len(label_names)
    print(f"Number of classes: {num_labels}")
    print(f"Classes: {label_names}")
    
    return dataset, label_names, num_labels

def preprocess_images(examples, processor):
    """Preprocess images for ViT"""
    images = [Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path 
              for image_path in examples["image"]]
    
    inputs = processor(images=images, return_tensors="pt")
    
    examples["pixel_values"] = inputs["pixel_values"]
    return examples

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_model():
    """Main training function"""
    # Load data
    dataset, label_names, num_labels = load_data()
    
    # Initialize processor
    print(f"\nLoading processor: {BASE_MODEL}")
    processor = ViTImageProcessor.from_pretrained(BASE_MODEL)
    
    # Preprocess datasets
    print("\nPreprocessing images...")
    train_dataset = dataset["train"].map(
        lambda x: preprocess_images(x, processor),
        batched=True,
        remove_columns=["image"]
    )
    
    val_dataset = dataset["validation"].map(
        lambda x: preprocess_images(x, processor),
        batched=True,
        remove_columns=["image"]
    )
    
    # Set format for PyTorch
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    
    # Initialize model
    print(f"\nLoading model: {BASE_MODEL}")
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_names)},
        label2id={label: i for i, label in enumerate(label_names)}
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=f"{MODEL_OUTPUT_DIR}/logs",
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Learning rate: {LEARNING_RATE}")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating on validation set...")
    metrics = trainer.evaluate()
    print(f"Final metrics: {metrics}")
    
    # Save final model
    print(f"\nSaving model to {MODEL_OUTPUT_DIR}")
    trainer.save_model(MODEL_OUTPUT_DIR)
    processor.save_pretrained(MODEL_OUTPUT_DIR)
    
    print("Training complete!")
    return MODEL_OUTPUT_DIR

if __name__ == "__main__":
    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Train
    model_path = train_model()
    print(f"\nModel saved at: {model_path}")
