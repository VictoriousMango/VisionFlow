# Image Preprocessing, OCR, Workflow Analysis, and Dataset Generation App

This project consists of two main components: `app.py`, a Streamlit application for image preprocessing, OCR, and workflow analysis, and `DatasetGenerator.py`, a Streamlit application for generating and managing datasets for computer vision tasks. Together, they form a pipeline for processing and analyzing images (e.g., flowcharts) and creating annotated datasets in COCO format for machine learning.

## Overview of `app.py`

The `app.py` Streamlit application processes images to extract text and analyze workflows, combining computer vision, optical character recognition (OCR), and AI-driven pseudocode generation. It allows users to upload an image, apply preprocessing techniques, extract text using OCR, and generate pseudocode for workflows using the Groq API.

### Functionality of `app.py`
- **Image Upload**: Users can upload images (JPG, PNG, JPEG) via a web interface.
- **Image Preprocessing**: Enhances images for better text extraction using techniques like grayscale conversion, thresholding, morphological opening, and edge detection.
- **OCR**: Extracts text from the original and preprocessed images using Pytesseract.
- **Workflow Analysis**: Analyzes the extracted text (from the thresholded image) using the Groq API to generate structured pseudocode for workflows.
- **Visualization**: Displays the original image, preprocessed images in a 2x2 grid, extracted text, and generated pseudocode.

## Components of `app.py`

### 1. Environment Setup
- **Purpose**: Loads sensitive configuration details securely.
- **Details**:
  - Uses `python-dotenv` to load variables from a `.env` file.
  - Retrieves `TESSERACT_PATH` (path to Tesseract executable, if needed) and `GROQ_API_KEY` (for Groq API authentication).
  - Sets the Tesseract command path if provided.
- **Libraries**: `dotenv`, `os`.

### 2. `ImagePreprocessor` Class
- **Purpose**: Handles image preprocessing to improve OCR accuracy.
- **Methods**:
  - `get_grayscale`: Converts the image to grayscale.
  - `remove_noise`: Applies median blur to reduce noise.
  - `thresholding`: Uses Otsu's thresholding for binary image conversion.
  - `dilate` and `erode`: Morphological operations to enhance image features.
  - `get_opening`: Combines erosion and dilation to remove small noise.
  - `canny`: Detects edges using the Canny algorithm.
  - `deskew`: Corrects image skew for better text alignment.
  - `match_template`: Template matching (not used in the main flow but available).
  - `preprocess`: Runs key preprocessing steps (grayscale, thresholding, opening, canny) and returns results as a dictionary.
- **Libraries**: `cv2` (OpenCV), `numpy`.

### 3. `OCRProcessor` Class
- **Purpose**: Extracts text from images using Pytesseract.
- **Methods**:
  - `__init__`: Initializes with a custom Tesseract configuration (`--oem 3 --psm 6` for general text extraction).
  - `extract_text`: Extracts text from a single image.
  - `process_images`: Extracts text from multiple images (original and preprocessed).
- **Libraries**: `pytesseract`.

### 4. `analyze_workflow_with_groq` Function
- **Purpose**: Analyzes extracted text to generate pseudocode for workflows using the Groq API.
- **Details**:
  - Authenticates with the Groq API using the `GROQ_API_KEY`.
  - Sends the extracted text to the `llama-3.3-70b-versatile` model with a prompt to generate structured pseudocode.
  - Handles errors (e.g., missing API key, network issues) gracefully.
- **Libraries**: `requests`.

### 5. `matplotlib_to_pil` Function
- **Purpose**: Converts Matplotlib figures to PIL images for Streamlit display.
- **Details**: Saves the figure to a buffer and converts it to a PIL Image.
- **Libraries**: `matplotlib.pyplot`, `PIL`, `io`.

### 6. `main` Function
- **Purpose**: Orchestrates the Streamlit app's workflow.
- **Details**:
  - Creates a Streamlit interface with a title and file uploader for images.
  - Displays the original image in RGB format.
  - Initializes `ImagePreprocessor` and `OCRProcessor`.
  - Preprocesses the uploaded image and displays results (grayscale, thresholded, opening, canny) in a 2x2 grid.
  - Extracts text from the original and preprocessed images, displaying each in a text area.
  - Analyzes the thresholded image's text using the Groq API and displays the generated pseudocode.
  - Uses spinners to indicate processing and warnings for empty text.
- **Libraries**: `streamlit`, `cv2`, `numpy`, `PIL`, `matplotlib.pyplot`, `uuid`.

## Overview of `DatasetGenerator.py`

The `DatasetGenerator.py` Streamlit application is designed to create and manage datasets for computer vision tasks, specifically for generating annotated flowchart images and converting them into COCO format for machine learning model training (e.g., object detection or image segmentation).

### Functionality of `DatasetGenerator.py`
- **Dataset Generation**: Creates a specified number of flowchart images with corresponding JSON annotations using a custom `GenerateDataset` class.
- **Annotation Export**: Converts annotations to a CSV file for easy inspection.
- **COCO Format Conversion**: Transforms JSON annotations into COCO format using a `J2C` (JSON2COCO) class.
- **Train-Validation Split**: Splits the dataset into training and validation sets in COCO format.
- **Visualization**: Displays generated images, JSON annotations, CSV annotations, COCO format data, and train/validation splits in a tabbed interface.
- **Dataset Management**: Allows users to clear the dataset directory to start fresh.

## Components of `DatasetGenerator.py`

### 1. `EmptyDirectory` Function
- **Purpose**: Clears all files in the `dataset/` directory and the `annotations.csv` file.
- **Details**: Uses `glob` to find files and `os.remove` to delete them, with a Streamlit spinner for feedback.
- **Libraries**: `glob`, `os`, `streamlit`.

### 2. Sidebar Controls
- **Purpose**: Provides user inputs for dataset generation and management.
- **Details**:
  - `DatasetNum`: A number input to specify how many flowchart images to generate.
  - `DatasetIndex`: A number input to select a specific dataset image/annotation for display.
  - `Start Pipeline` Button: Triggers the dataset generation pipeline.
  - `Empty Datasets` Button: Calls `EmptyDirectory` to clear the dataset.
- **Libraries**: `streamlit`.

### 3. Pipeline Execution
- **Purpose**: Runs the dataset generation and processing pipeline.
- **Details**:
  - **Dataset Creation**: Uses `GenerateDataset.CreateDataset` to generate `DatasetNum` flowchart images and JSON annotations.
  - **CSV Conversion**: Uses `GenerateDataset.ScriptToCSV` to export annotations to `annotations.csv`.
  - **COCO Conversion**: Uses `J2C.convert_to_coco` to create `instances_coco.json` in COCO format.
  - **Train-Validation Split**: Uses `J2C.TrainValSplit` to create `instances_train.json` and `instances_val.json`.
- **Dependencies**: Custom `GenerateDataset` and `J2C` classes (assumed in `Assets/` directory).
- **Libraries**: `streamlit`.

### 4. Tabbed Interface
- **Purpose**: Displays dataset details in an organized manner.
- **Tabs**:
  - **Dataset**:
    - Shows the selected flowchart image (`flowchart_XXX.jpg`) and its JSON annotation (`flowchart_XXX.json`).
    - Handles errors if files are missing.
  - **Annotations**:
    - Displays `annotations.csv` as a DataFrame if it exists.
  - **COCO Format Dataset**:
    - Shows `instances_coco.json` data split into images, annotations, and categories as DataFrames.
  - **Train Validation Split**:
    - Sub-tabs for `instances_train.json` and `instances_val.json`, each showing images, annotations, and categories as DataFrames.
- **Libraries**: `streamlit`, `pandas`, `json`.

## Dependencies
### For `app.py`
- Python 3.8+
- Libraries: `streamlit`, `opencv-python`, `numpy`, `pytesseract`, `Pillow`, `matplotlib`, `python-dotenv`, `requests`.
- External: Tesseract-OCR installed (with path in `.env` if non-standard), Groq API key.

### For `DatasetGenerator.py`
- Python 3.8+
- Libraries: `streamlit`, `pandas`, `json`, `os`, `glob`, `subprocess`.
- Custom Modules: `Assets.DatasetGenerator.GenerateDataset`, `Assets.JSON2COCO.J2C`.
- Directory: Expects a `dataset/` folder for output.

## Usage
### For `app.py`
1. Create a `.env` file with:
   ```env
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe  # Optional
   GROQ_API_KEY=your-groq-api-key
   ```
2. Install dependencies:
   ```bash
   pip install streamlit opencv-python numpy pytesseract Pillow matplotlib python-dotenv requests
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Upload an image to preprocess, extract text, and view pseudocode.

### For `DatasetGenerator.py`
1. Ensure the `Assets/` directory contains `DatasetGenerator.py` and `JSON2COCO.py`.
2. Install dependencies:
   ```bash
   pip install streamlit pandas
   ```
3. Run the app:
   ```bash
   streamlit run DatasetGenerator.py
   ```
4. Use the sidebar to specify the number of datasets, start the pipeline, and view results in the tabs.

## Notes
- **Integration**: `app.py` can process images generated by `DatasetGenerator.py` (e.g., `flowchart_XXX.jpg`) for OCR and workflow analysis, creating a cohesive pipeline.
- **Security**: Ensure `.env` is not committed to version control (add to `.gitignore`).
- **Error Handling**: Both apps handle missing files or API errors with warnings in the Streamlit interface.
- **Custom Modules**: `DatasetGenerator.py` relies on `GenerateDataset` and `J2C`, which must be implemented correctly in `Assets/`.