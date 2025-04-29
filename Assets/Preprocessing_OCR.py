import re
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import os

# Image preprocessing functions (unchanged)
class Preprocess():
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def dilate(self, image):
        kernel = np.ones((5,5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def erode(self, image):
        kernel = np.ones((5,5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def get_opening(self, image):
        kernel = np.ones((5,5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def canny(self, image):
        return cv2.Canny(image, 100, 200)

    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def match_template(self, image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    def preProcessIMG(self, img):
        # Read the image
        image = cv2.imread(img)
        b, g, r = cv2.split(image)
        rgb_img = cv2.merge([r, g, b])
        plt.imshow(rgb_img)
        plt.title('ORIGINAL IMAGE')
        plt.show()

        # Preprocess image
        gray = self.get_grayscale(image)
        thresh = self.thresholding(gray)
        opening = self.get_opening(gray)
        canny = self.canny(gray)

        images = {'gray': gray, 'thresh': thresh, 'opening': opening, 'canny': canny}

        # Plot images after preprocessing
        fig = plt.figure(figsize=(13,13))
        ax = []
        rows = 2
        columns = 2
        keys = list(images.keys())
        for i in range(rows*columns):
            ax.append(fig.add_subplot(rows, columns, i+1))
            ax[-1].set_title('AUREBESH - ' + keys[i])
            plt.imshow(images[keys[i]], cmap='gray')
        plt.show()
'''
# Get OCR output using Pytesseract
custom_config = r'--oem 3 --psm 6'

# Store the extracted text from different preprocessing steps
extracted_texts = {}

print('-----------------------------------------')
print('TESSERACT OUTPUT --> ORIGINAL IMAGE')
print('-----------------------------------------')
extracted_texts['original'] = pytesseract.image_to_string(image, config=custom_config)
print(extracted_texts['original'])

print('\n-----------------------------------------')
print('TESSERACT OUTPUT --> THRESHOLDED IMAGE')
print('-----------------------------------------')
extracted_texts['thresh'] = pytesseract.image_to_string(thresh, config=custom_config)
print(extracted_texts['thresh'])

print('\n-----------------------------------------')
print('TESSERACT OUTPUT --> OPENED IMAGE')
print('-----------------------------------------')
extracted_texts['opening'] = pytesseract.image_to_string(opening, config=custom_config)
print(extracted_texts['opening'])

print('\n-----------------------------------------')
print('TESSERACT OUTPUT --> CANNY EDGE IMAGE')
print('-----------------------------------------')
extracted_texts['canny'] = pytesseract.image_to_string(canny, config=custom_config)
print(extracted_texts['canny'])

# Let's assume the thresholded image gives the best result for now
best_extracted_text = extracted_texts['thresh']
'''

if __name__=="__main__":
    path = "../dataset/flowchart_000.jpg"
    obj = Preprocess()
    obj.preProcessIMG(path)