import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pytesseract
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.utils import to_categorical
from IPython.display import display
from ipywidgets import FileUpload
import io
import os
upload = FileUpload(accept='image/*', multiple=False)
display(upload)



def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to a fixed size (e.g., 128x128)
    resized = cv2.resize(gray, (128, 128))
    
    # Normalize the image to 0-1
    normalized = resized / 255.0
    
    # Optionally apply GaussianBlur or other denoising techniques if needed
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    
    return blurred
def upload_image(upload):
    if upload.value:
        print(upload.value[0],'---')
        # Extract image after upload
        for fileinfo in upload.value:
            image_bytes = fileinfo['content']
            filename = fileinfo['name']
            print(image_bytes,'-----hello')
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            saved_imagepath = os.path.join("F:/workspace/image_reader/dataset", filename) 
            image.save(saved_imagepath)  # Save to disk if needed
            print(saved_imagepath,"saved_imagepath")
            print(f"Image '{filename}' uploaded successfully.")
            return saved_imagepath


try:
    # image_path = "F:\workspace\card1.png"
    image_path =upload_image(upload)
    processed_image = preprocess_image(image_path)
    print('image progress success')
    
except Exception as e:
    print(e,'image processing')

def create_ocr_model(input_shape=(128, 128, 1)):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))

    model.add(layers.Reshape((16, 64)))  # (batch_size, 16, 64)
    model.add(layers.TimeDistributed(layers.Dense(36, activation='softmax')))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model




def encode_load_data(image_paths, labels):
    print(image_paths,labels,'check')
    images = []
    labels_encoded = []

    for image_path, label in zip(image_paths, labels):
        image = preprocess_image(image_path)
        images.append(image)

        label = label.replace(" ", "")  # Remove spaces
        label_onehot_sequence = []

        for char in label:
            onehot = np.zeros(36)
            if char.isdigit():
                index = int(char)
            else:
                index = ord(char.upper()) - ord('A') + 10
            onehot[index] = 1
            label_onehot_sequence.append(onehot)

        labels_encoded.append(label_onehot_sequence)

    images = np.array(images)
    images = np.expand_dims(images, axis=-1)  # Shape: (num_samples, 128, 128, 1)
    labels_encoded = np.array(labels_encoded)  # Shape: (num_samples, 16, 36)
    print(images,labels_encoded,'encoded load data check')
    return images, labels_encoded

def load_data(image_dir, label_dir, image_size=(128, 32)):
    images = []
    labels = []
    
    for img_name in os.listdir(image_dir):
        # if img_name.endswith(".jpg"):  # Adjust the format if necessary
            # Load image
            img_path = os.path.join(image_dir, img_name)
            # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # img = cv2.resize(img, image_size)
            # img = img / 255.0  # Normalize to [0, 1]
            images.append(img_path)
            
            # Load corresponding label
            label_path = os.path.join(label_dir, img_name.replace(".png", ".txt").replace(".jpg", ".txt").replace(".svg", ".txt").replace(".jpeg", ".txt"))
            with open(label_path, 'r') as f:
                label = f.read().strip()
            labels.append(label)
    
    # images = np.array(images)
    # images = np.expand_dims(images, axis=-1)  # Add channel dimension for grayscale
    
    return images, labels

try:
    ocr_model = create_ocr_model()
    ocr_model.summary()
    print('model success')
except Exception as e:
    print(e,'create model error')

# image_paths = ['F:/workspace/image_reader/dataset/card1.png', 'F:/workspace/image_reader/dataset/card2.png', 'F:/workspace/image_reader/dataset/card3.png']
# labels = ['4000 1234 5678 9010', '4000 1234 5678 9010', '5412 7512 3412 3456']

try:
    image_dir = os.path.join('F:/workspace/image_reader/dataset','images')
    label_dir = os.path.join('F:/workspace/image_reader/dataset','labels')
    x, y = load_data(image_dir, label_dir)
    print(x,'xxxx')
    print(y,'yyyy')
    train_images, train_labels = encode_load_data(x, y)
    # print(train_images,train_labels,'train image and label success')
except Exception as e:
    print(e,'images and label error')

# Train the model
try:
    ocr_model.fit(train_images, train_labels, epochs=10, batch_size=1)
    model_dir ="F:/workspace/image_reader"
    modelpath=os.path.join(model_dir, "ocrmodel.h5")

    ocr_model.save(modelpath)
    print('model saved successfully')
except Exception as e:
    print(e,'ocr model train the model error')

def predict_account_number(image_path):
    processed_image = preprocess_image(image_path)
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image = np.expand_dims(processed_image, axis=-1)

    prediction = ocr_model.predict(processed_image)  # Shape: (1, 16, 36)
    print(prediction,"prediction")
    predicted_digits = []

    for i in range(16):
        pred_index = np.argmax(prediction[0][i])
        if pred_index < 10:
            predicted_digits.append(str(pred_index))
        else:
            predicted_digits.append(chr(ord('A') + pred_index - 10))

    return ''.join(predicted_digits)


# Test the model with a new image
try:
    # image_path = "F:\workspace\card1.png"
    image_path = upload_image(upload)
    # image_path = 'F:\workspace\test_debit_card_image.jpg'
    account_number = predict_account_number(image_path)
    print("Predicted Account Number:", account_number)
except Exception as e:
    print(e,'predict')
