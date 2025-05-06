import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import io
import uuid
import requests

# Load environment variables
load_dotenv()
tesseract_path = os.getenv("TESSERACT_PATH")
groq_api_key = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


class ImagePreprocessor:
    """Class to handle image preprocessing steps."""

    @staticmethod
    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def remove_noise(image):
        return cv2.medianBlur(image, 5)

    @staticmethod
    def thresholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    @staticmethod
    def dilate(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    @staticmethod
    def erode(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    @staticmethod
    def get_opening(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def canny(image):
        return cv2.Canny(image, 100, 200)

    @staticmethod
    def deskew(image):
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

    @staticmethod
    def match_template(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    def preprocess(self, image):
        """Apply all preprocessing steps and return results."""
        gray = self.get_grayscale(image)
        thresh = self.thresholding(gray)
        opening = self.get_opening(gray)
        canny = self.canny(gray)
        return {
            'gray': gray,
            'thresh': thresh,
            'opening': opening,
            'canny': canny
        }


class OCRProcessor:
    """Class to handle OCR processing with Pytesseract."""

    def __init__(self, config=r'--oem 3 --psm 6'):
        self.config = config

    def extract_text(self, image):
        """Extract text from an image using Pytesseract."""
        return pytesseract.image_to_string(image, config=self.config)

    def process_images(self, images):
        """Extract text from multiple preprocessed images."""
        extracted_texts = {}
        for key, img in images.items():
            extracted_texts[key] = self.extract_text(img)
        return extracted_texts


def analyze_workflow_with_groq(text):
    """Analyze workflow text using Groq API to generate pseudocode."""
    if not groq_api_key:
        return "Error: GROQ_API_KEY not found in .env file."

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",  # Updated to a valid Groq model
        "messages": [
            {"role": "system",
             "content": "You are an expert in analyzing workflow diagrams and generating python code. Extract the sequence of steps, decisions, and actions from the provided text and output structured python code."},
            {"role": "user", "content": f"Analyze this workflow text and generate python code: {text}"}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: Failed to connect to Groq API - {str(e)}"


def matplotlib_to_pil(fig):
    """Convert Matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)


def main():
    st.title("Image Preprocessing, OCR, and Workflow Analysis App")
    st.write("Upload an image to preprocess it, extract text using OCR, and analyze workflows with Groq AI.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display original image
        st.subheader("Original Image")
        b, g, r = cv2.split(image)
        rgb_img = cv2.merge([r, g, b])
        st.image(rgb_img, use_column_width=True)

        # Initialize processors
        preprocessor = ImagePreprocessor()
        ocr_processor = OCRProcessor()

        # Preprocess image
        with st.spinner("Preprocessing image..."):
            preprocessed_images = preprocessor.preprocess(image)

        # Display preprocessed images
        st.subheader("Preprocessed Images")
        fig = plt.figure(figsize=(10, 10))
        rows, columns = 2, 2
        keys = list(preprocessed_images.keys())
        for i in range(rows * columns):
            ax = fig.add_subplot(rows, columns, i + 1)
            ax.set_title(keys[i].upper())
            plt.imshow(preprocessed_images[keys[i]], cmap='gray')
            plt.axis('off')
        st.image(matplotlib_to_pil(fig), use_column_width=True)
        plt.close(fig)

        # Extract text
        with st.spinner("Extracting text..."):
            extracted_texts = ocr_processor.process_images({
                'original': image,
                **preprocessed_images
            })

        # Display extracted text
        st.subheader("Extracted Text")
        for key, text in extracted_texts.items():
            st.write(f"**{key.upper()}**:")
            st.text_area(f"Text from {key}", text, height=150, key=str(uuid.uuid4()))

        # Analyze workflow with Groq
        st.subheader("Workflow Analysis (Pseudocode)")
        with st.spinner("Analyzing workflow with Groq AI..."):
            best_extracted_text = extracted_texts.get('thresh', '')
            if best_extracted_text.strip():
                pseudocode = analyze_workflow_with_groq(best_extracted_text)
                st.text_area("Generated Pseudocode", pseudocode, height=200, key="pseudocode")
            else:
                st.warning("No text extracted from thresholded image to analyze.")


if __name__ == "__main__":
    main()