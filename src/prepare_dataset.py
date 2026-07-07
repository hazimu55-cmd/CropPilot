"""
Alternative approach: Use existing model with better filtering
Since public datasets are having issues, we'll use a different strategy:
1. Use the existing pre-trained model as base
2. Create a custom dataset loader for your specific images
3. Fine-tune on your collected data
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json

# Configuration
DATA_DIR = "data/custom_dataset"
OUTPUT_DIR = "data/processed_dataset"

def create_custom_dataset_structure():
    """
    Create directory structure for custom dataset
    User should organize images as:
    data/custom_dataset/
        Corn/
            healthy/
                img1.jpg
            disease1/
                img1.jpg
        Potato/
            healthy/
            disease1/
        etc.
    """
    crops = ["Corn", "Potato", "Rice", "Wheat", "Cotton"]
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for crop in crops:
        os.makedirs(os.path.join(DATA_DIR, crop, "healthy"), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, crop, "diseased"), exist_ok=True)
    
    print(f"Created dataset structure at {DATA_DIR}")
    print("\nPlease organize your images as follows:")
    print(f"{DATA_DIR}/")
    for crop in crops:
        print(f"  {crop}/")
        print(f"    healthy/")
        print(f"    diseased/")
    
    print("\nAfter organizing images, run: python src/train_custom.py")

class CustomCropDataset(Dataset):
    """Custom dataset for crop disease images"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        self.images = []
        self.labels = []
        self.label_names = []
        
        # Load all images and create labels
        label_map = {}
        current_label = 0
        
        for crop in os.listdir(data_dir):
            crop_path = os.path.join(data_dir, crop)
            if not os.path.isdir(crop_path):
                continue
            
            for condition in os.listdir(crop_path):
                condition_path = os.path.join(crop_path, condition)
                if not os.path.isdir(condition_path):
                    continue
                
                label_name = f"{crop}_{condition}"
                if label_name not in label_map:
                    label_map[label_name] = current_label
                    self.label_names.append(label_name)
                    current_label += 1
                
                for img_file in os.listdir(condition_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(condition_path, img_file))
                        self.labels.append(label_map[label_name])
        
        # Split data
        if split == 'train':
            self.images, _, self.labels, _ = train_test_split(
                self.images, self.labels, test_size=0.2, random_state=42, stratify=self.labels
            )
        elif split == 'val':
            _, self.images, _, self.labels = train_test_split(
                self.images, self.labels, test_size=0.2, random_state=42, stratify=self.labels
            )
        
        print(f"{split} set: {len(self.images)} images")
        print(f"Classes: {self.label_names}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == "__main__":
    create_custom_dataset_structure()
