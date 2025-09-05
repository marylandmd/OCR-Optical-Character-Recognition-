# TÌm Email trong ảnh
import re
import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
plt.rcParams["figure.figsize"] = (20, 30)

email_pattern = '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
image_file =r"C:\Users\Cun\Downloads\1131w-wPpAXSlmfF4.webp"
# Đọc ảnh
img = cv2.imread(image_file)

d = pytesseract.image_to_data(img, output_type=Output.DICT)
# print('\n', d.keys(), '\n')

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        if re.match(email_pattern, d['text'][i]):
            (x, y, w, h) = d["left"][i], d["top"][i], d["width"][i], d["height"][i] 
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) #Plotting bounding box
            print(f"Email: {d['text'][i]}")
        
plt.imshow(img)
