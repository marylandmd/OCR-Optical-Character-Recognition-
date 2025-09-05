# Create green boxes around texts in image

import pytesseract
from pytesseract import Output
import cv2
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

plt.rcParams["figure.figsize"] = (20, 30)
image_file =r"C:\Users\Cun\Downloads\1131w-wPpAXSlmfF4.webp"

# Read image
img = cv2.imread(image_file)
d = pytesseract.image_to_data(img, output_type=Output.DICT)

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60 and d['text'][i].strip() != "":
        (x, y, w, h) = d["left"][i], d["top"][i], d["width"][i], d["height"][i] 
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # hộp màu xanh lá
        img = cv2.putText(img, d['text'][i], (x, y-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # chữ màu đỏ


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show boxes 
plt.imshow(img_rgb)
plt.axis("off")
plt.show()


