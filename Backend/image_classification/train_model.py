import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

print("TensorFlow version:", tf.__version__)

# Load the CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("Dataset loaded successfully!")
print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")

# --- Data Augmentation Section ---
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

# Fit the data augmentation generator on the training data
datagen.fit(train_images)
# --- End of Data Augmentation Section ---

print("Building and training the model...")

# Build a simple (CNN) with a slightly deeper architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu')) # Added a new layer
model.add(layers.MaxPooling2D((2, 2)))

# Add dense layers for classification
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu')) # Increased neurons
model.add(layers.Dense(10))  # 10 output neurons for 10 classes

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


print("Model architecture:")
model.summary()

# Train the model with the augmented data for more epochs
print("Starting training with data augmentation...")
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=30,  # Increased epochs for better learning
                    validation_data=(test_images, test_labels),
                    verbose=1)

# Evaluate the model's performance
print("Evaluating model...")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"\nModel accuracy on test data: {test_acc:.4f}")

# Save the trained model to a file
model_path = '../cifar10_image_classifier.h5'
model.save(model_path)
print(f"Model saved to {model_path}")

# Check if file was created
if os.path.exists(model_path):
    print("Model file verification: SUCCESS")
else:
    print("Model file verification: FAILED")

print("Training completed successfully!")