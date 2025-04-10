import cv2
import pytesseract


class Extractor:

    def __init__(self, img):
        # Load image and crop a detected shape (example coordinates)
        self.img = cv2.imread(img)

    def readRectangle(self, coordinates):
        # print("Called Read Rectangle")
        x, y, w, h = coordinates #218, 166, 76, 94  # Replace with CNN output
        cropped = self.img[y:y + h, x:x + w]

        # Preprocess (grayscale, threshold)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Run OCR
        text = pytesseract.image_to_string(thresh)
        # print("Detected text:", text)
        return text


if __name__ == "__main__":
    shapes = [
        [295, 345, 63, 88],
        [237, 142, 81, 91],
        [51, 74, 65, 74],
    ]
    img = "synthetic_flowchart.jpg"

    # Pipeline
    obj = Extractor(img)
    extracted_Text=[]
    for shape in shapes:
            extracted_Text.append(obj.readRectangle(shape))

    for text in extracted_Text:
        print(f"Extracted Text : {text}")