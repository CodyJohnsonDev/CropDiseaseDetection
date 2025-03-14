import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb  # Optional for tracking experiments
import multiprocessing
import torch.backends.cudnn as cudnn

# Ensure proper multiprocessing start method on macOS
if __name__ == '__main__':
    # On macOS, 'spawn' is more reliable than the default 'fork'
    multiprocessing.set_start_method('spawn', force=True)

# Enable cuDNN benchmarking for faster training
cudnn.benchmark = True

# Check for Apple Silicon and use MPS if available
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS acceleration on Apple Silicon")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA acceleration")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# Define directories and parameters
TRAIN_DIR = "PlantVillage/train"
VAL_DIR = "PlantVillage/val"
IMG_HEIGHT = 224  # Reduced from 384 for faster training
IMG_WIDTH = 224
BATCH_SIZE = 64  # Increased batch size for faster training
NUM_WORKERS = 4  # Parallel data loading
EPOCHS = 30
LEARNING_RATE = 1e-4  # Reduced from 3e-4 to prevent NaN issues
WEIGHT_DECAY = 0.01  # Reduced from 0.05 to prevent NaN issues
MODEL_NAME = "efficientnet_b0"  # Lighter model than ConvNeXt
GRADIENT_ACCUMULATION_STEPS = 2  # Simulate larger batch sizes
GRAD_CLIP = 1.0  # Add gradient clipping to prevent exploding gradients

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define dataset class
class PlantDiseaseDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Filter out hidden files like .DS_Store
        self.classes = [d for d in sorted(os.listdir(root_dir)) 
                       if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self.get_image_paths()
    
    def get_image_paths(self):
        image_paths = []
        labels = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            # Skip files that aren't directories
            if not os.path.isdir(class_path):
                continue
                
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                # Skip directories and hidden files
                if os.path.isdir(img_path) or img_name.startswith('.'):
                    continue
                # Skip non-image files
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                image_paths.append(img_path)
                labels.append(self.class_to_idx[class_name])
        return list(zip(image_paths, labels))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the label
            placeholder = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            if self.transform:
                augmented = self.transform(image=placeholder)
                placeholder = augmented["image"]
            return placeholder, label

# Simpler transforms to reduce computational load
train_transform = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Training and validation functions
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()  # Zero gradients at the beginning
    
    pbar = tqdm(loader, desc="Training")
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            # Gradient clipping to prevent NaN
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Compute metrics (use the original loss value, not the scaled one)
        batch_loss = loss.item() * accumulation_steps * inputs.size(0)
        running_loss += batch_loss
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Check for NaN loss and print relevant information
        if torch.isnan(loss).any():
            print(f"NaN loss detected at batch {i}!")
            print(f"Inputs min/max: {inputs.min().item()}/{inputs.max().item()}")
            print(f"Outputs: {outputs[:5]}")  # Print first 5 outputs
            # Break the loop to avoid further NaN propagation
            break
        
        pbar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })
    
    # Make sure to call optimizer.step() for the last batch if needed
    if len(loader) % accumulation_steps != 0:
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
    
    # Calculate metrics
    epoch_loss = running_loss / total if total > 0 else float('nan')
    epoch_acc = 100. * correct / total if total > 0 else 0
    
    if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
        f1 = f1_score(all_labels, all_preds, average='weighted')
    else:
        f1 = 0.0
    
    return epoch_loss, epoch_acc, f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Inference
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Compute metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
    
    # Calculate metrics
    epoch_loss = running_loss / total if total > 0 else float('nan')
    epoch_acc = 100. * correct / total if total > 0 else 0
    
    if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
        f1 = f1_score(all_labels, all_preds, average='weighted')
    else:
        f1 = 0.0
    
    return epoch_loss, epoch_acc, f1, all_preds, all_labels

# Main execution block
if __name__ == '__main__':
    try:
        # Create datasets and dataloaders
        train_dataset = PlantDiseaseDataset(TRAIN_DIR, transform=train_transform)
        val_dataset = PlantDiseaseDataset(VAL_DIR, transform=val_transform)
        
        print(f"Found {len(train_dataset.classes)} classes: {train_dataset.classes}")
        print(f"Training set: {len(train_dataset)} images")
        print(f"Validation set: {len(val_dataset)} images")
        
        # Check for empty dataset
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty. Check your directory paths.")
        
        # Check for balanced classes
        class_counts = [0] * len(train_dataset.classes)
        for _, label in train_dataset.images:
            class_counts[label] += 1
        
        print("Class distribution:")
        for i, (cls, count) in enumerate(zip(train_dataset.classes, class_counts)):
            print(f"  {cls}: {count} images")
        
        # Standard dataloaders without weighted sampling for now
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,  # Simple shuffling instead of weighted sampling
            num_workers=NUM_WORKERS,
            pin_memory=True if DEVICE != torch.device("cpu") else False,
            drop_last=False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True if DEVICE != torch.device("cpu") else False,
            drop_last=False,
        )
        
        # Create the model using a lighter architecture
        model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=len(train_dataset.classes),
            drop_rate=0.2,  # Dropout for regularization
        )
        
        # Initialize weights with careful scaling
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Move model to device
        model = model.to(DEVICE)
        
        # Define loss function (without label smoothing to start)
        criterion = nn.CrossEntropyLoss()
        
        # Use AdamW optimizer with reduced weight decay
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Learning rate scheduler (simpler than before)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Training loop
        best_val_acc = 0.0
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            
            # Train
            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE,
                grad_clip=GRAD_CLIP, accumulation_steps=GRADIENT_ACCUMULATION_STEPS
            )
            
            # Check for NaN loss
            if np.isnan(train_loss):
                print("NaN loss detected during training. Stopping training.")
                # Save the current model state for debugging
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'classes': train_dataset.classes,
                }, "debug_model_with_nan_loss.pth")
                break
            
            # Validate
            val_loss, val_acc, val_f1, val_preds, val_labels = validate(
                model, val_loader, criterion, DEVICE
            )
            
            # Update learning rate based on validation accuracy
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'classes': train_dataset.classes,
                }, "best_plant_disease_model.pth")
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # After training, plot the history if training completed
        if len(history['train_loss']) > 0:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history['train_loss'], label='Train')
            plt.plot(history['val_loss'], label='Validation')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot(history['train_acc'], label='Train')
            plt.plot(history['val_acc'], label='Validation')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            plt.plot(history['train_f1'], label='Train')
            plt.plot(history['val_f1'], label='Validation')
            plt.title('F1 Score')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            print("Training history plot saved as 'training_history.png'")
        
        print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    
    except Exception as e:
        import traceback
        print(f"An error occurred during training: {e}")
        traceback.print_exc()