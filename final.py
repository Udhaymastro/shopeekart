from django.http import HttpResponse,JsonResponse
from django.shortcuts import redirect,render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import pandas as pd
import cv2
import easyocr
import re
import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
# import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.utils import to_categorical
from IPython.display import display
from ipywidgets import FileUpload
import io
from .trainmodel import detect_text_regions, recognize_text,predict_account_number

reader = easyocr.Reader(['en'],gpu=False)

def mask_card_numbers(img_path):
    image = cv2.imread(img_path)
    results = reader.readtext(image)

    digit_blocks = []
    for bbox, text, conf in results:
        cleaned = text.replace(" ", "").replace("-", "")
        if cleaned.isdigit() and 3 <= len(cleaned) <= 4:
            digit_blocks.append((bbox, cleaned))

    digit_blocks.sort(key=lambda x: x[0][0][0])
    masked_path = img_path.replace(".jpg", "_masked.jpg").replace(".png", "_masked.png").replace(".jpeg", "_masked.jpeg")
    if len(digit_blocks) >= 4:
        to_mask = digit_blocks[:-1]
        for bbox, _ in to_mask:
            pts = np.array(bbox, dtype=np.int32)
            cv2.fillPoly(image, [pts], (0, 0, 0))
        cv2.imwrite(masked_path, image)

    return os.path.basename(img_path), os.path.basename(masked_path)

def uploadpage(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_path = fs.path(filename)
        
        input_img, output_img = mask_card_numbers(uploaded_path)

        return render(request, 'account.html', {
            'input_img': fs.url(input_img),
            'output_img': fs.url(output_img),
        })

    return render(request, 'account.html')



def pretrainedmodel(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        img_path = fs.save(image.name, image)
        full_path = fs.path(img_path)

        boxes, image = detect_text_regions(full_path)
        digit_blocks =[]
        print(boxes,"boxes")
        for (x1, y1, x2, y2), crop in boxes:
            text = recognize_text(crop)
            cleaned = text.replace(" ", "").replace("-", "")
            if cleaned.isdigit() and 3 <= len(cleaned) <= 4:
                digit_blocks.append(((x1, y1, x2, y2), cleaned))
        to_mask = digit_blocks[:-1]
        for (x1, y1, x2, y2), _ in to_mask:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)


        output_path = os.path.join(fs.location, f"masked_{img_path}")
        cv2.imwrite(output_path, image)

        return render(request, 'account.html', {
            'inputmodel_img': fs.url(img_path),
            'outputmodel_img': fs.url(f"masked_{img_path}")
        })

def my_trained_model(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        fs = FileSystemStorage()
        img_path = fs.save(image.name, image)
        image_path = fs.path(img_path)
        output_path = os.path.join(fs.location, f"masked_{img_path}")
        # masked_img = mask_account_number(image_path,output_path)
        account_number = predict_account_number(image_path)
        image = cv2.imread(image_path)  # Replace with your image path
        image_height, image_width = image.shape[:2]

        # Your model output (assuming it's a numpy array)
        model_output = np.array(account_number)  # Replace with actual variable
        # boxes = model_output[0]  # Assuming shape: (1, num_boxes, num_features)
        print(model_output,"model_output")
        # Loop through each detection
        for box in model_output:

            x_center, y_center, width, height = box[:4]
            x_center = int(x_center * image_width)
            y_center = int(y_center * image_height)
            width = int(width * image_width)
            height = int(height * image_height)

            # Calculate box corners
            x1 = max(0, x_center - width // 2)
            y1 = max(0, y_center - height // 2)
            x2 = min(image_width, x_center + width // 2)
            y2 = min(image_height, y_center + height // 2)
            print(x1, y1, x2, y2)
        # print(account_number,"account_number")

        # boxes, image = detect_text_regions(full_path)
        # digit_blocks =[]

        # for (x1, y1, x2, y2), crop in boxes:
        #     text = recognize_text(crop)
        #     cleaned = text.replace(" ", "").replace("-", "")
        #     if cleaned.isdigit() and 3 <= len(cleaned) <= 4:
        #         digit_blocks.append(((x1, y1, x2, y2), cleaned))
        # to_mask = digit_blocks[:-1]
        # for (x1, y1, x2, y2), _ in to_mask:
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)


        # output_path = os.path.join(fs.location, f"masked_{img_path}")
        # cv2.imwrite(output_path, image)

        return render(request, 'account.html', {
            'input_image': fs.url(image_path),
            'output_image': fs.url(f"masked_{masked_img}")
        })

def mask_account_number(image_path, save_path):
    image = cv2.imread(image_path)


    start_x = 100    
    start_y = 250  
    digit_width = 40
    digit_height = 50

    for i in range(16):
        x1 = start_x + i * digit_width
        y1 = start_y
        x2 = x1 + digit_width
        y2 = y1 + digit_height
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

    cv2.imwrite(save_path, image)
    return save_path


@csrf_exempt
def home(request):
    return render(request,'account.html')




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

# try:
#     ocr_model = create_ocr_model()
#     ocr_model.summary()
#     print('model success')
# except Exception as e:
#     print(e,'create model error')

# image_paths = ['F:/workspace/image_reader/dataset/card1.png', 'F:/workspace/image_reader/dataset/card2.png', 'F:/workspace/image_reader/dataset/card3.png']
# labels = ['4000 1234 5678 9010', '4000 1234 5678 9010', '5412 7512 341 2 3456']

# try:
#     image_dir = os.path.join('dataset','images')
#     label_dir = os.path.join('dataset','labels')
#     x, y = load_data(image_dir, label_dir)
#     print(x,'xxxx')
#     print(y,'yyyy')
#     train_images, train_labels = encode_load_data(x, y)
#     # print(train_images,train_labels,'train image and label success')
# except Exception as e:
#     print(e,'images and label error')

# # Train the model
# try:
#     ocr_model.fit(train_images, train_labels, epochs=10, batch_size=1)
#     # model_dir ="F:/workspace/image_reader"
#     modelpath=os.path.join("ocrmodel.h5")

#     ocr_model.save(modelpath)
#     print('model saved successfully')
# except Exception as e:
#     print(e,'ocr model train the model error')

# def predict_account_number(image_path):
#     processed_image = preprocess_image(image_path)
#     processed_image = np.expand_dims(processed_image, axis=0)
#     processed_image = np.expand_dims(processed_image, axis=-1)

#     prediction = ocr_model.predict(processed_image)  # Shape: (1, 16, 36)
#     print(prediction,"prediction")
#     predicted_digits = []

#     for i in range(16):
#         pred_index = np.argmax(prediction[0][i])
#         if pred_index < 10:
#             predicted_digits.append(str(pred_index))
#         else:
#             predicted_digits.append(chr(ord('A') + pred_index - 10))

#     return ''.join(predicted_digits)


# # Test the model with a new image
# try:
#     # image_path = "F:\workspace\card1.png"
#     image_path = upload_image(upload)
#     # image_path = 'F:\workspace\test_debit_card_image.jpg'
#     account_number = predict_account_number(image_path)
#     print("Predicted Account Number:", account_number)
# except Exception as e:
#     print(e,'predict')
