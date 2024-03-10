import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image

# Step 1: Preprocess the custom image
def preprocess_custom_image(image_path):
    img = Image.open(image_path)
    
    img = img.resize((28, 28))
    
    # Convert the image to grayscale
    img = img.convert('L')
    
    # Normalize the pixel values to the range [0, 1]
    img = np.array(img) / 255.0
    
    # Ensure the shape matches the expected input shape of the model
    img = np.expand_dims(img, axis=-1)  # Add an extra dimension for channels
    
    return img
def main():
    # Step 2: Load the Fashion MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Step 3: Train a machine learning model
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

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    train_images = np.expand_dims(train_images, axis=-1) / 255.0
    test_images = np.expand_dims(test_images, axis=-1) / 255.0

    model.fit(train_images, train_labels, epochs=3, batch_size=64, validation_data=(test_images, test_labels))

    # Step 4: Use the trained model to classify the custom image
    custom_image_path = "/Users/krishangratra/Documents/Coding/Hackathons/Women Techies '24/cloth-detector/images/pic12.jpg"

    custom_image = preprocess_custom_image(custom_image_path)
    custom_image = np.expand_dims(custom_image, axis=0) / 255.0

    predicted_probabilities = model.predict(custom_image)

    # Print the predicted probabilities for each class
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Loose Fit/Baggy', 'Ankle boot']
    for i, prob in enumerate(predicted_probabilities[0]):
        # print(f"Probability of class {class_names[i]}: {prob:.4f}")
        continue

    print(predicted_probabilities)

    # Get the predicted class index
    predicted_class = np.argmax(predicted_probabilities)

    # Get the indices of the top 3 classes with highest probabilities
    top_3_indices = np.argsort(-predicted_probabilities)[0][:4]

    # Print the top 3 classes and their probabilities
    preference_list = []
    for i in top_3_indices:
        if i == 5:
            if (predicted_probabilities[0][i] > 0.94):
                str1 = f'{class_names[i]}'
                preference_list.append(str1)
                print(f"Class: {class_names[i]}, Probability: {predicted_probabilities[0][i]:.4f}")
        else:
            str1 = f'{class_names[i]}'
            preference_list.append(str1)
            print(f"Class: {class_names[i]}, Probability: {predicted_probabilities[0][i]:.4f}")


    # Print the predicted class label
    # predicted_label = class_names[predicted_class]
    # print("Predicted Label:", predicted_label)

    # Step 5: Evaluate model performance on the test set
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(preference_list)
    return preference_list

main()
