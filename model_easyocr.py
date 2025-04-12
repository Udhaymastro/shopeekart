import cv2
import easyocr
import re
import os
import re
import cv2
import numpy as np


image_path = os.path.join("F:/workspace/image_reader/dataset/images","card1.png") 

image = cv2.imread(image_path)

reader = easyocr.Reader(['en'],gpu=False)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

# Detect and blur credit card-like numbers
images, gray = preprocess_image(image_path)
results = reader.readtext(images)
def mask_region(img, bbox):
    pts = np.array(bbox, dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))  # black fill
    return img


digit_blocks = []

for bbox, text, conf in results:
    cleaned = text.replace(" ", "").replace("-", "")
    if cleaned.isdigit() and 3 <= len(cleaned) <= 4:  # assuming credit card parts like '1234'
        digit_blocks.append((bbox, cleaned))

# Sort from left to right by x-coordinate
digit_blocks.sort(key=lambda x: x[0][0][0])  # sort by top-left x

# Mask all except the last 4 digits
if len(digit_blocks) >= 4:
    to_mask = digit_blocks[:-1]  # leave only last block visible
    for bbox, text in to_mask:
        print(f"Masking: {text}")
        image = mask_region(image, bbox)
# Save or display the result
cv2.imwrite("masked_card.jpg", image)

