import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

# Define dataset paths
DATASET_DIR = "PlantVillage"  # Adjust if needed
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Function to count images per class
def count_images(folder):
    class_counts = {}
    for cls in os.listdir(folder):
        class_path = os.path.join(folder, cls)
        if os.path.isdir(class_path):  # Ensure it's a directory
            class_counts[cls] = len([
                f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))
            ])
    return class_counts

# Get counts for train and validation sets
train_counts = count_images(TRAIN_DIR)
val_counts = count_images(VAL_DIR)

# Create DataFrame with train/val image counts
df = pd.DataFrame({"Train": train_counts, "Validation": val_counts}).fillna(0)
df = df.astype(int)  # Ensure counts are integers

# Save dataset summary
df.to_csv("dataset_summary.csv")
print("Dataset summary saved as 'dataset_summary.csv'")
print(df)

# Plot Train vs. Validation class distribution
plt.figure(figsize=(14, 6))
df.plot(kind="bar", figsize=(14, 6), title="Train vs. Validation Counts")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Crop Disease Category")
plt.ylabel("Image Count")
plt.legend(["Train", "Validation"])
plt.tight_layout()

# Save plot
plt.savefig("dataset_distribution.png")
print("Dataset distribution plot saved as 'dataset_distribution.png'")
plt.show()

# Function to display random images from a random class
def display_sample_images():
    random_class = random.choice(list(train_counts.keys()))
    class_dir = os.path.join(TRAIN_DIR, random_class)
    sample_images = random.sample(os.listdir(class_dir), min(5, len(os.listdir(class_dir))))

    print(f"Displaying sample images from class: {random_class}")
    plt.figure(figsize=(12, 6))
    
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(class_dir, img_name)
        try:
            img = Image.open(img_path)
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.title(random_class)
            plt.axis("off")
        except Exception as e:
            print(f"Error opening image {img_name}: {e}")
    
    plt.tight_layout()
    plt.show()

# Uncomment to display sample images
display_sample_images()
