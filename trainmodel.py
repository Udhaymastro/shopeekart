import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
# import pytesseract
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.utils import to_categorical
from IPython.display import display
from ipywidgets import FileUpload
import io
import os


from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
upload = FileUpload(accept='image/*', multiple=False)
display(upload)



def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    normalized = resized / 255.0
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




def predict_account_number(image_path):
    processed_image = preprocess_image(image_path)
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image = np.expand_dims(processed_image, axis=-1)
    
    MODEL_PATH = os.path.join('maskapp', 'ocrmodel.h5')
    model = load_model(MODEL_PATH, compile=False)
    prediction = model.predict(processed_image)
    predicted_digits = []

    for i in range(16):
        pred_index = np.argmax(prediction[0][i])
        if pred_index < 10:
            predicted_digits.append(str(pred_index))
        else:
            predicted_digits.append(chr(ord('A') + pred_index - 10))

    # return ''.join(predicted_digits)

    return predicted_digits


# # Test the model with a new image
# try:
#     # image_path = "F:\workspace\card1.png"
#     image_path = upload_image(upload)
#     # image_path = 'F:\workspace\test_debit_card_image.jpg'
#     account_number = predict_account_number(image_path)
#     print("Predicted Account Number:", account_number)
# except Exception as e:
#     print(e,'predict')



################################Pretarined model#####################################


processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-stage1')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-stage1')
MODEL_PATH = os.path.join('maskapp', 'frozen_east_text_detection.pb')
print(MODEL_PATH,'MODEL_PATH')

def detect_text_regions(image_path):
    net = cv2.dnn.readNet(MODEL_PATH)
    print(net,"net")
    image = cv2.imread(image_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    newW, newH = (320, 320)
    rW, rH = W / newW, H / newH
    resized = cv2.resize(image, (newW, newH))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH), (123.68,116.78,103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    results = []
    for (startX, startY, endX, endY) in boxes:
        x1, y1, x2, y2 = int(startX*rW), int(startY*rH), int(endX*rW), int(endY*rH)
        roi = orig[y1:y2, x1:x2]
        results.append(((x1, y1, x2, y2), roi))
    return results, orig

def decode_predictions(scores, geometry, min_confidence=0.5):
    (numRows, numCols) = scores.shape[2:4]
    rects, confidences = [], []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos, sin = np.cos(angle), np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX, startY = int(endX - w), int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))
    return rects, confidences

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0: return []

    boxes = boxes.astype("float")
    pick = []

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2 if probs is None else probs)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:-1]]
        idxs = np.delete(idxs, np.concatenate(([len(idxs)-1], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def recognize_text(image_array):
    pil_image = Image.fromarray(image_array).convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(inputs)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text
