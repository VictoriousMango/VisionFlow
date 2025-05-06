# VisionFlow: Workflow Diagram Analyzer

This project, `VisionFlow`, is a Streamlit application designed to analyze workflow diagrams (e.g., flowcharts, ER diagrams) by processing images to extract text, detect shapes and connections, classify workflows, and generate structured outputs like SQL schemas or Python code. It leverages computer vision, optical character recognition (OCR), and AI-driven analysis to provide insights into workflow diagrams.

## Overview

`FlowChart Analyzer 2.py` is a Streamlit application that processes images of workflow diagrams to:
- Extract text using OCR.
- Detect shapes and connections in the diagram.
- Classify the diagram as either a database schema or procedural logic.
- Generate SQL DDL statements for database schemas or Python code for procedural workflows.
- Provide AI-enhanced analysis using the Groq API.
- Visualize results with annotated images and structured outputs.

The application supports JPG, PNG, and JPEG image uploads and provides a user-friendly interface to view preprocessing steps, detected components, extracted text, and generated code.

## Functionality

- **Image Upload**: Users can upload workflow diagram images (JPG, PNG, JPEG) via a web interface.
- **Image Preprocessing**: Enhances images for better OCR and shape detection using techniques like grayscale conversion, thresholding, morphological operations, and edge detection.
- **OCR**: Extracts text from shapes within the diagram using Pytesseract.
- **Shape and Connection Detection**: Identifies shapes (e.g., rectangles, circles, diamonds) and their connections (e.g., arrows, lines) to understand the diagram’s structure.
- **Workflow Classification**: Classifies the diagram as a database schema (e.g., ER diagram) or procedural logic (e.g., flowchart) based on shape types and text content.
- **Code Generation**:
  - For database schemas: Generates SQL `CREATE TABLE` statements with columns and relationships.
  - For procedural workflows: Generates Python code with proper control flow (e.g., conditionals, loops).
- **AI-Enhanced Analysis**: Uses the Groq API to generate enhanced Python code based on extracted text and diagram structure.
- **Visualization**: Displays the original image, annotated diagram with detected shapes and connections, extracted text, and generated code/SQL in a tabbed interface.

## Components

### 1. Environment Setup
- **Purpose**: Loads sensitive configuration details securely.
- **Details**:
  - Uses `python-dotenv` to load variables from a `.env` file.
  - Retrieves `TESSERACT_PATH` (path to Tesseract executable, if non-standard) and `GROQ_API_KEY` (for Groq API authentication).
  - Sets the Tesseract command path if provided.
- **Libraries**: `dotenv`, `os`.

### 2. `ImagePreprocessor` Class
- **Purpose**: Preprocesses images to improve OCR and shape detection accuracy.
- **Methods**:
  - `get_grayscale`: Converts the image to grayscale.
  - `remove_noise`: Applies median blur to reduce noise.
  - `thresholding`: Uses Otsu’s thresholding for binary image conversion.
  - `adaptive_threshold`: Applies adaptive thresholding for shape detection.
  - `dilate` and `erode`: Morphological operations to enhance image features.
  - `get_opening` and `get_closing`: Combines erosion and dilation to remove noise or fill gaps.
  - `canny`: Detects edges using the Canny algorithm.
  - `preprocess`: Runs the preprocessing pipeline and returns a dictionary of processed images (grayscale, thresholded, adaptive, opening, closing, canny, binary).
- **Libraries**: `cv2` (OpenCV), `numpy`.

### 3. `OCRProcessor` Class
- **Purpose**: Extracts text from images and regions of interest (ROIs) using Pytesseract.
- **Methods**:
  - `__init__`: Initializes with a custom Tesseract configuration (`--oem 3 --psm 6` for general text extraction).
  - `extract_text`: Extracts text from a single image.
  - `extract_text_from_roi`: Extracts text from a specific region (e.g., inside a shape).
  - `process_images`: Extracts text from multiple preprocessed images.
- **Libraries**: `pytesseract`.

### 4. `ShapeDetector` Class
- **Purpose**: Detects and classifies shapes in the workflow diagram.
- **Methods**:
  - `__init__`: Sets a minimum area threshold to filter out noise.
  - `detect_shapes`: Finds contours, approximates polygons, classifies shapes (e.g., rectangle, circle, diamond), and extracts text from each shape using OCR.
  - `classify_shape`: Classifies shapes based on the number of vertices (e.g., 4 for rectangle, >6 for circle).
- **Libraries**: `cv2`, `pytesseract`.

### 5. `ConnectionDetector` Class
- **Purpose**: Detects connections (e.g., arrows, lines) between shapes.
- **Methods**:
  - `detect_connections`: Uses Hough Line Transform to identify lines, masks out shapes, and maps lines to start/end shapes.
  - `find_closest_shape`: Finds the shape closest to a given point for connection mapping.
- **Libraries**: `cv2`, `numpy`.

### 6. `VisualizationUtils` Class
- **Purpose**: Visualizes detected shapes and connections.
- **Methods**:
  - `visualize_detection`: Draws contours, bounding boxes, and connection lines with arrowheads on the original image.
  - `matplotlib_to_pil`: Converts Matplotlib figures to PIL images for Streamlit display.
- **Libraries**: `cv2`, `matplotlib.pyplot`, `PIL`, `io`.

### 7. `WorkflowClassifier` Class
- **Purpose**: Classifies the diagram as a database schema or procedural logic.
- **Methods**:
  - `classify_workflow`: Uses heuristics (shape type ratios, keyword analysis) to determine the workflow type with a confidence score.
- **Libraries**: None (uses standard Python).

### 8. `SQLGenerator` Class
- **Purpose**: Generates SQL DDL statements for database schema diagrams.
- **Methods**:
  - `generate_sql_schema`: Identifies tables from rectangles, extracts columns and data types from text, and creates `CREATE TABLE` statements with primary and foreign key constraints.
- **Libraries**: None (uses standard Python).

### 9. `CodeGenerator` Class
- **Purpose**: Generates Python code for procedural workflow diagrams.
- **Methods**:
  - `generate_python_code`: Builds a graph from shapes and connections, sorts shapes by y-coordinate, and generates Python code with proper control flow (e.g., `if` statements for decisions, `print` for outputs).
- **Libraries**: `re`.

### 10. `AIAnalyzer` Class
- **Purpose**: Enhances workflow analysis using the Groq API.
- **Methods**:
  - `analyze_workflow_with_groq`: Sends extracted text, shapes, and connections to the Groq API (`llama-3.3-70b-versatile` model) to generate structured Python code with error handling.
- **Libraries**: `requests`.

### 11. `main` Function
- **Purpose**: Orchestrates the Streamlit app’s workflow.
- **Details**:
  - Creates a Streamlit interface with a title and file uploader.
  - Displays the original image in RGB format.
  - Initializes all processing classes (`ImagePreprocessor`, `OCRProcessor`, etc.).
  - Processes the uploaded image: preprocesses, detects shapes/connections, classifies the workflow, and generates SQL or Python code.
  - Displays results in tabs: detection results (annotated image, shapes), generated code/SQL, AI-enhanced code, and extracted text.
  - Provides download buttons for generated code/SQL and AI-enhanced code.
- **Libraries**: `streamlit`, `cv2`, `numpy`, `PIL`, `matplotlib.pyplot`, `uuid`.

## Dependencies
- **Python**: 3.8+
- **Libraries**:
  - `streamlit`: For the web interface.
  - `opencv-python`: For image processing.
  - `numpy`: For numerical operations.
  - `pytesseract`: For OCR.
  - `Pillow`: For image handling.
  - `matplotlib`: For visualization.
  - `python-dotenv`: For environment variable management.
  - `requests`: For Groq API calls.
  - `scikit-image`: For image measurement utilities.
- **External**:
  - Tesseract-OCR installed (with path in `.env` if non-standard).
  - Groq API key (required for AI analysis).

## Usage
1. **Set up the environment**:
   Create a `.env` file in the project root with:
   ```env
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe  # Optional, if non-standard
   GROQ_API_KEY=your-groq-api-key
   ```
   Ensure `.env` is not committed to version control (add to `.gitignore`).

2. **Install dependencies**:
   ```bash
   pip install streamlit opencv-python numpy pytesseract Pillow matplotlib python-dotenv requests scikit-image
   ```

3. **Run the app**:
   ```bash
   streamlit run FlowChart Analyzer 2.py
   ```

4. **Use the app**:
   - Open the Streamlit interface in your browser (typically `http://localhost:8501`).
   - Upload a workflow diagram image (JPG, PNG, JPEG).
   - View the processed results in the tabs:
     - **Detection Results**: Annotated image with shapes and connections, workflow type, and shape details.
     - **SQL Schema/Python Code**: Generated SQL or Python code with a download button.
     - **AI Enhanced Analysis**: AI-generated Python code with a download button.
     - **Extracted Text**: Raw text extracted from the diagram.
   - Download generated outputs as needed.

## Notes
- **Error Handling**: The app handles missing files, API errors, or empty text with Streamlit warnings and error messages.
- **Performance**: Large or complex diagrams may require more processing time; spinners indicate progress.
- **Customization**: The preprocessing pipeline, shape classification, and code generation can be extended by modifying the respective classes.
- **Security**: Ensure the Groq API key is kept secure and not exposed in the codebase.
- **Limitations**:
  - OCR accuracy depends on image quality and text clarity.
  - Shape detection may miss small or overlapping shapes (adjust `min_area` in `ShapeDetector` if needed).
  - AI analysis requires a valid Groq API key and internet connection.
