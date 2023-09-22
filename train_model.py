import tensorflow as tf
from keras import layers, models
import os
from keras.preprocessing.image import ImageDataGenerator

# Define paths and variables
dataset_path = os.path.join('.', 'merged_dataset')  # Use the merged dataset path
test_path = os.path.join('.', 'test_dataset')  # Replace with your test dataset path
train_path = os.path.join(dataset_path, 'train')  # Update the train path
val_path = os.path.join(dataset_path, 'validate')  # Update the validation path

# List subfolders (gesture classes)
gesture_classes = os.listdir(train_path)  # Use the train path to list classes

# Create a dictionary to map class names to numerical labels
class_to_label = {cls: i for i, cls in enumerate(gesture_classes)}

# Define image dimensions and batch size
image_size = (64, 64)
batch_size = 32

# Create data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Rescale pixel values to [0, 1]
    rotation_range=20,  # Augmentation parameters (adjust as needed)
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only rescale for validation data

# Flow training and validation images from directories
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse' for integer labels
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse' for integer labels
    shuffle=False  # Do not shuffle validation data
)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(gesture_classes), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Save the trained model to a file
model.save("model_v1.h5")

# Evaluate the model on the test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Rescale pixel values to [0, 1] for test data
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse' for integer labels
    shuffle=False  # Do not shuffle test data
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')