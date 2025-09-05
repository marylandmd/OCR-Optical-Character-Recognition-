import re
import cv2
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_file = r"C:\Users\Cun\Downloads\8bf41652-955e-49eb-aa15-8e461ca64efe.jpg"  # ảnh Cetaphil

# 1) Preprocessing: zoom + contrast boost + light binary
img0 = cv2.imread(image_file)
scale = 1.8
img = cv2.resize(img0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
L2 = clahe.apply(L)
img_enh = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

gray = cv2.cvtColor(img_enh, cv2.COLOR_BGR2GRAY)
thr  = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 31, 9)

# 2) Full text OCR
config = "--oem 1 --psm 6"  # LSTM + page segmentation theo dòng
raw_text = pytesseract.image_to_string(thr, lang="eng", config=config)

#3) Find the flexible 'INGREDIENTS', take the following part on the same line 
# (+1 line next, accept 'ingredients' / 'ingredient' )
m = re.search(r'\bingredients?\b\s*[:\-–]?\s*(.+)', raw_text, flags=re.IGNORECASE)
ingredients_text = None

if m:
    first_line = m.group(1).strip()
    after = raw_text[m.end():].splitlines()
    cont = ""
    if after:
        nxt = after[0].strip()
        if nxt and not re.match(r'^\s*(directions?|trusted|defends|skin|distributed|galderma|made in|lot)\b', nxt, re.I):
            cont = nxt
    ingredients_text = (first_line + (" " + cont if cont else "")).strip(" .;,:")
else:
    ingredients_text = "Ingredients section not found."

print("INGREDIENTS:")
print(ingredients_text)

#4) display image used OCR ----
plt.figure(figsize=(8, 12))
plt.imshow(thr, cmap='gray')
plt.axis('off'); plt.title("Image used for OCR")
plt.show()
