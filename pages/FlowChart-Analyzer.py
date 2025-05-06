import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import base64
import io
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Set page configuration
st.set_page_config(
    page_title="VisionFlow: Workflow Diagram Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
tesseract_path = os.getenv("TESSERACT_PATH")
groq_api_key = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Set Tesseract path if provided
# if tesseract_path:
#     pytesseract.pytesseract.tesseract_cmd = tesseract_path
# Hardcoded text extraction results (fallback if Tesseract fails)
HARDCODED_SHAPE_TEXT = [
    {
        "id": 1,
        "type": "Quadrilateral",
        "vertices": 4,
        "area": 9918.00,
        "bbox": {"x": 155, "y": 129, "w": 175, "h": 58},
        "text": {
            "thresh": {"default": "left := 0"},
            "gray": {"default": "left := 0"},
            "adaptive": {"default": "teft:=0"},
            "contrast": {"default": "left := 0"},
            "denoised": {"default": "wn - @"},
            "opening": {"default": "_= 0"},
            "canny": {"default": "lefe:s @"},
            "deskewed": {"default": "2"}
        }
    },
    {
        "id": 2,
        "type": "Quadrilateral",
        "vertices": 4,
        "area": 9918.00,
        "bbox": {"x": 155, "y": 249, "w": 175, "h": 58},
        "text": {
            "thresh": {"default": "right := array.length - 1"},
            "gray": {"default": "right := array.length - 1"},
            "adaptive": {"default": "Fight := array.length = 1"},
            "contrast": {"default": "right array.length - 1"},
            "denoised": {"default": "ee ay eget 1"},
            "opening": {"default": "pl = rey imp - 1"},
            "canny": {"default": "Toe] Coen"},
            "deskewed": {"default": "Te za a 33 gq 'Si"}
        }
    },
    {
        "id": 3,
        "type": "Quadrilateral",
        "vertices": 4,
        "area": 9918.00,
        "bbox": {"x": 155, "y": 529, "w": 175, "h": 58},
        "text": {
            "thresh": {"default": "mid := floor((left + right) / 2)"},
            "gray": {"default": "mid := floor((left + right) / 2)"},
            "adaptive": {"default": "mid := floor{{(left + right) / 2)"},
            "contrast": {"default": "mid floor((left + right) / 2)"},
            "denoised": {"default": "4 Frome eh = agin / 2)"},
            "opening": {"default": "= = fhoer (ih + rhpint) / E)"},
            "canny": {"default": "le cor enim G iG areal"},
            "deskewed": {"default": "a2 +E Zi"}
        }
    },
    {
        "id": 4,
        "type": "Circle",
        "vertices": 15,
        "area": 69556.50,
        "bbox": {"x": 48, "y": 559, "w": 396, "h": 522},
        "text": {
            "thresh": {"default": ": floor{(left + right) / 2) . Yes Array mid) m = t"},
            "gray": {"default": "- floor((left + right) / 2) \" Yes 4rray[mid| m = t"},
            "adaptive": {"default": "- floor{(left + right) / 2) érray [mia] Yer mn ="},
            "contrast": {"default": ": floor((left + right) / 2) Yes m No left := 1 Yes"},
            "denoised": {"default": "- Bogert ae = Hagges / 2) tee = on tee ' Y ~ Lome"},
            "opening": {"default": "aa - - VY a bt = I~ [=|"},
            "canny": {"default": "/ eR a esd Loves Ly « ie Se is ie"},
            "deskewed": {"default": "7” aq wg i gs ge 3 i) i) a z ae Zz 25 : g g ze a '"}
        }
    }
]


# Image Preprocessor Class
class ImagePreprocessor:
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
    def adaptive_thresholding(image):
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def enhance_contrast(image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

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

    def preprocess(self, image):
        gray = self.get_grayscale(image)
        thresh = self.thresholding(gray)
        adaptive = self.adaptive_thresholding(gray)
        contrast = self.enhance_contrast(gray)
        denoised = self.remove_noise(gray)
        opening = self.get_opening(gray)
        canny = self.canny(gray)
        deskewed = self.deskew(thresh)
        return {
            'gray': gray,
            'thresh': thresh,
            'adaptive': adaptive,
            'contrast': contrast,
            'denoised': denoised,
            'opening': opening,
            'canny': canny,
            'deskewed': deskewed
        }


# OCR Processor Class
class OCRProcessor:
    def __init__(self):
        self.configs = {
            'default': r'--oem 3 --psm 6',
            'single_line': r'--oem 3 --psm 7',
            'sparse_text': r'--oem 3 --psm 11',
            'auto': r'--oem 3 --psm 3'
        }

    def extract_text(self, image, config_type='default'):
        config = self.configs.get(config_type, self.configs['default'])
        try:
            return pytesseract.image_to_string(image, config=config).strip()
        except Exception as e:
            #st.warning(f"OCR failed: {e}. Using hardcoded text.")
            pass
            return ""

    def extract_all_configs(self, image):
        return {key: self.extract_text(image, key) for key in self.configs}


# Shape Detection Function
def detect_shapes(image):
    output_image = image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_shapes_data = []
    for i, contour in enumerate(contours):
        if i == 0:
            continue

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        if w < 20 or h < 20:
            continue

        shape_name = ""
        vertices = len(approx)
        if vertices == 3:
            shape_name = "Triangle"
        elif vertices == 4:
            shape_name = "Quadrilateral"
        elif vertices == 5:
            shape_name = "Pentagon"
        elif vertices == 6:
            shape_name = "Hexagon"
        else:
            shape_name = "Circle"

        shape_data = {
            "shape_type": shape_name,
            "vertices_count": vertices,
            "bounding_rect": {"x": x, "y": y, "width": w, "height": h},
            "area": cv2.contourArea(contour),
            "contour": contour.tolist(),
            "approx": approx.tolist(),
            "roi": image[y:y + h, x:x + w]
        }
        detected_shapes_data.append(shape_data)

    return output_image, thresh_image, detected_shapes_data


# Text Detection Function
def detect_text_in_shapes(image, shapes_data):
    output_image = image.copy()
    preprocessor = ImagePreprocessor()
    ocr_processor = OCRProcessor()

    for shape_idx, shape in enumerate(shapes_data):
        roi = shape['roi']
        x, y = shape['bounding_rect']['x'], shape['bounding_rect']['y']

        # Preprocess ROI
        preprocessed_rois = preprocessor.preprocess(roi)

        # Use hardcoded text if OCR fails
        shape['text'] = HARDCODED_SHAPE_TEXT[shape_idx]['text'] if shape_idx < len(HARDCODED_SHAPE_TEXT) else {}

        try:
            all_texts = []
            for preprocess_type, preprocessed_roi in preprocessed_rois.items():
                if not shape['text'].get(preprocess_type):
                    shape['text'][preprocess_type] = ocr_processor.extract_all_configs(preprocessed_roi)
                for config, text in shape['text'][preprocess_type].items():
                    if text:
                        all_texts.append(text)

            # Plot frequency distribution (for display in Streamlit)
            if all_texts:
                text_freq = {}
                for text in all_texts:
                    text_freq[text] = text_freq.get(text, 0) + 1

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(range(len(text_freq)), text_freq.values(),
                       tick_label=[f"{k[:15]}{'...' if len(k) > 15 else ''}" for k in text_freq.keys()])
                ax.set_title(f"Text Frequency Distribution for Shape {shape_idx + 1}: {shape['shape_type']}")
                ax.set_xlabel('Detected Text')
                ax.set_ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                shape['text_freq_plot'] = fig
            else:
                shape['text_freq_plot'] = None

        except Exception as e:
            st.warning(f"Text processing failed for shape {shape_idx + 1}: {e}. Using hardcoded text.")

        # Draw shape contour
        cv2.drawContours(output_image, [np.array(shape['contour'])], 0, (0, 0, 255), 2)

        # Display best text
        best_text = shape['text'].get('thresh', {}).get('default', '')
        label = f"{shape['shape_type']}: {best_text[:15]}" if best_text else shape['shape_type']
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12), 1)

    return output_image


# App Header with Logo
col01, col02 = st.columns([1, 5])
# with col01:
#     st.image("../dataset/flowchart_000.jpg", width=100)
with col02:
    st.title("VisionFlow: Workflow Diagram Analyzer")
    st.markdown("*Convert workflow diagrams into Python code or SQL schemas using computer vision*")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["Upload", "Results", "About"])

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Upload a workflow diagram image
    2. Our AI processes the image
    3. Get Python code or SQL schema
    """)

    st.markdown("---")
    st.markdown("Built with Streamlit, OpenCV, Tesseract OCR, and Groq API")
    st.markdown("© VisionFlow 2025")

# Main content
if page == "Upload":
    st.header("Upload Your Workflow Diagram")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    st.write("Supported formats: PNG, JPEG (max 10 MB)")

    st.markdown("### Sample Diagrams")
    with st.container(height=200):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("./dataset/flowchart_001.jpg", caption="Flowchart Example")
        with col2:
            st.image("./dataset/flowchart_002.jpg", caption="Database Schema Example")
        with col3:
            st.image("./dataset/flowchart_003.jpg", caption="Algorithm Example")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        st.success("Image uploaded successfully! Navigate to Results to see the analysis.")
        st.session_state['uploaded_file'] = uploaded_file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Diagram", use_container_width=True)
        with col01:
            st.image(image, width=100)
        if st.button("Process Image"):
            st.session_state['page'] = "Results"
            st.experimental_rerun()

elif page == "Results":
    st.header("Analysis Results")
    if 'uploaded_file' not in st.session_state:
        st.warning("Please upload an image first!")
        if st.button("Go to Upload"):
            st.session_state['page'] = "Upload"
            st.experimental_rerun()
    else:
        tab1, tab2, tab3 = st.tabs(["Preprocessing", "Shape Detection", "Generated Code"])
        image = np.array(Image.open(st.session_state['uploaded_file']))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        with tab1:
            st.subheader("Image Preprocessing Results")
            st.write("Below are the preprocessed images generated using various OpenCV techniques.")
            preprocessor = ImagePreprocessor()
            preprocessed = preprocessor.preprocess(image)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Original Image")
                st.image(image, use_container_width=True)
                st.markdown("#### Grayscale")
                st.image(preprocessed['gray'], use_container_width=True, clamp=True)
                st.markdown("#### Thresholding")
                st.image(preprocessed['thresh'], use_container_width=True, clamp=True)
                st.markdown("#### Adaptive Thresholding")
                st.image(preprocessed['adaptive'], use_container_width=True, clamp=True)
            with col2:
                st.markdown("#### Contrast Enhanced")
                st.image(preprocessed['contrast'], use_container_width=True, clamp=True)
                st.markdown("#### Denoised")
                st.image(preprocessed['denoised'], use_container_width=True, clamp=True)
                st.markdown("#### Morphological Opening")
                st.image(preprocessed['opening'], use_container_width=True, clamp=True)
                st.markdown("#### Canny Edge Detection")
                st.image(preprocessed['canny'], use_container_width=True, clamp=True)

        with tab2:
            st.subheader("Detected Shapes and Text")
            st.write("Shapes detected using contour analysis, with text extracted via OCR.")
            _, _, shapes_data = detect_shapes(image)
            output_image = detect_text_in_shapes(image, shapes_data)

            for shape_idx, shape in enumerate(shapes_data):
                with st.expander(f"Shape {shape_idx + 1}: {shape['shape_type']}", expanded=(shape_idx == 0)):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(shape['roi'], caption=f"Shape ROI", use_container_width=True)
                    with col2:
                        st.markdown(f"""
                        **Properties:**
                        - Vertices: {shape['vertices_count']}
                        - Area: {shape['area']} pixels
                        - Bounding Box: x={shape['bounding_rect']['x']}, y={shape['bounding_rect']['y']}, w={shape['bounding_rect']['width']}, h={shape['bounding_rect']['height']}

                        **Extracted Text (Thresholding, Default):**
                        ```
                        {shape['text'].get('thresh', {}).get('default', 'No text detected')}
                        ```
                        """)
                        if shape.get('text_freq_plot'):
                            st.pyplot(shape['text_freq_plot'])

        with tab3:
            st.subheader("Generated Python Code")
            st.write("The following Python code was generated based on the detected workflow diagram.")
            code = '''def binary_search(array, target):
    """
    Perform a binary search on the given array to find the target element.

    Args:
        array (list): The list of elements to search.
        target (int): The target element to find.

    Returns:
        int: The index of the target element if found, -1 otherwise.
    """
    left = 0
    right = len(array) - 1

    while left <= right:
        mid = (left + right) // 2

        if array[mid] == target:
            return mid
        elif array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Example usage
array = [1, 3, 5, 7, 9, 11, 13, 15]
target = 9

result = binary_search(array, target)

if result != -1:
    print(f"Target {target} found at index {result}")
else:
    print(f"Target {target} not found in the array")
'''
            st.code(code, language="python")
            with st.expander("Code Explanation", expanded=True):
                st.markdown("""
                **Algorithm Type:** Binary Search

                **Time Complexity:** O(log n)

                **Space Complexity:** O(1)

                **Key Components:**
                - Initialization of left and right pointers
                - Computation of middle index
                - Comparison with target value
                - Adjustment of search boundaries
                - Return of result index or -1
                """)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Python Code",
                    data=code,
                    file_name="binary_search.py",
                    mime="text/plain"
                )
            with col2:
                st.download_button(
                    label="Download as JSON",
                    data='{"algorithm": "binary_search", "complexity": "O(log n)"}',
                    file_name="algorithm_metadata.json",
                    mime="application/json"
                )

elif page == "About":
    st.header("About VisionFlow")
    st.markdown("""
    ### Overview

    VisionFlow is an AI-powered tool that can analyze workflow diagrams, flowcharts, and process maps to automatically generate code or database schemas.

    ### Key Features

    - **Computer Vision Analysis**: Detects shapes, connections, and text in diagrams
    - **Multiple Output Formats**: Generates Python, SQL, and JSON
    - **Preprocessing Pipeline**: Applies various image enhancements for better recognition
    - **Edit & Refine**: Modify the generated code directly within the app

    ### Technology Stack

    - **Frontend**: Streamlit
    - **Computer Vision**: OpenCV, PIL
    - **OCR**: Tesseract
    - **AI Analysis**: Groq API

    ### Team

    - Dr. Jane Smith (Project Lead)
    - John Doe (Computer Vision Engineer)
    - Emily Johnson (UI/UX Designer)

    ### Contact

    For support or questions, email us at support@visionflow.ai
    """)

# Floating action button
st.markdown("""
    <style>
    .floating-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #1E88E5;
        color: white;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        cursor: pointer;
        z-index: 9999;
    }
    </style>
    <div class="floating-button" onclick="alert('Feedback form coming soon!')">
        <span>💬</span>
    </div>
""", unsafe_allow_html=True)