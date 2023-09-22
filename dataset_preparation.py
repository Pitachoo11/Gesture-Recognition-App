import os
import shutil
import random  # Add this line to import the random module

# Define paths
dataset_path = os.path.join('.', 'raw_dataset')  # Replace with your dataset path
merged_dataset_path = os.path.join('.', 'merged_dataset')  # Replace with your desired merged dataset path

# Create merged dataset folder
os.makedirs(merged_dataset_path, exist_ok=True)

# Define split ratio (e.g., 80% for training, 20% for validation)
split_ratio = 0.8

# Create train and validation folders
train_path = os.path.join(merged_dataset_path, 'train')
val_path = os.path.join(merged_dataset_path, 'validate')
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Initialize a set to keep track of copied files
copied_files = set()

# Initialize a list to keep track of empty class folders
empty_class_folders = []

# Iterate through the subfolders in the dataset
for root, _, _ in os.walk(dataset_path):
    if os.path.isdir(root):
        # Determine the class label from the class subfolder name
        class_label = os.path.basename(root)
        
        # Create class-specific train and validation folders if they don't exist
        class_train_path = os.path.join(train_path, class_label)
        class_val_path = os.path.join(val_path, class_label)
        os.makedirs(class_train_path, exist_ok=True)
        os.makedirs(class_val_path, exist_ok=True)
        
        # Collect the images in the class subfolder
        images = [os.path.join(root, file) for file in os.listdir(root) if file.endswith('.png')]
        
        # Calculate the number of images for training and validation
        num_images = len(images)
        num_train = int(num_images * split_ratio)
        
        # Randomly shuffle the images
        random.shuffle(images)
        
        # Split images into train and validation sets
        train_images = images[:num_train]
        val_images = images[num_train:]
        
        # Copy images to the respective train and validation class folders
        for image_path in train_images:
            # Check if the file has already been copied to avoid duplicates
            if image_path not in copied_files:
                shutil.copy(image_path, os.path.join(class_train_path, os.path.basename(image_path)))
                copied_files.add(image_path)
        for image_path in val_images:
            # Check if the file has already been copied to avoid duplicates
            if image_path not in copied_files:
                shutil.copy(image_path, os.path.join(class_val_path, os.path.basename(image_path)))
                copied_files.add(image_path)
        
        # Check if the class folders are empty
        if not os.listdir(class_train_path):
            empty_class_folders.append(class_train_path)
        if not os.listdir(class_val_path):
            empty_class_folders.append(class_val_path)

# Remove empty class-specific folders
for empty_folder in empty_class_folders:
    os.rmdir(empty_folder)

# Now, your dataset is split into train and validation sets, organized by class, and empty folders are removed.

def count_files_in_directory(directory):
    return sum([len(files) for _, _, files in os.walk(directory)])

# Count the number of files in each directory
raw_dataset_count = count_files_in_directory(dataset_path)
train_count = count_files_in_directory(train_path)
val_count = count_files_in_directory(val_path)

print(f"Total files in 'raw_dataset': {raw_dataset_count}")
print(f"Total files in 'train' folder: {train_count}")
print(f"Total files in 'validate' folder: {val_count}")