import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image

# Step 1: Preprocess the custom image
# Replace this with your custom image preprocessing code
def preprocess_custom_image(image_path):
    img = Image.open(image_path)
    
    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))
    
    # Convert the image to grayscale
    img = img.convert('L')
    
    # Normalize the pixel values to the range [0, 1]
    img = np.array(img) / 255.0
    
    # Ensure the shape matches the expected input shape of the model
    img = np.expand_dims(img, axis=-1)  # Add an extra dimension for channels
    
    return img

# Step 2: Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Step 3: Train a machine learning model
# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Preprocess and reshape the data
train_images = np.expand_dims(train_images, axis=-1) / 255.0
test_images = np.expand_dims(test_images, axis=-1) / 255.0

# Train the model
model.fit(train_images, train_labels, epochs=15, batch_size=64, validation_data=(test_images, test_labels))

# Step 4: Use the trained model to classify the custom image
# Replace 'custom_image' with your custom image data

custom_image = "/Users/krishangratra/Documents/Coding/Hackathons/Women Techies '24/Github Repos/test6/images/pic4.jpeg"

custom_image = preprocess_custom_image(custom_image)
custom_image = np.expand_dims(custom_image, axis=0) / 255.0

# Predict the class of the custom image
predicted_class = np.argmax(model.predict(custom_image))

# Step 5: Compare the classification result with the labels of the Fashion MNIST dataset
# Get the corresponding label from the Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
predicted_label = class_names[predicted_class]

print("Predicted Label:", predicted_label)
