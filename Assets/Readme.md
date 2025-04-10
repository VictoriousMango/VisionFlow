# Different shapes for flowcharts:

1. Rectangle: for processes
2. Diamond: for decisions
3. Oval/Circle: for start/end points
4. Parallelogram: for input/output operations

# Flowchart Generator and Dataset Creation

This Python script generates flowchart images and their corresponding JSON labels, which can be used for training object detection models. It utilizes OpenCV for image manipulation and drawing, and JSON for storing labeled data.

## Features

-   Generates flowchart images with various shapes (rectangle, diamond, oval, parallelogram).
-   Adds text inside shapes.
-   Draws arrows to connect flowchart elements.
-   Supports decision branches with "True" and "False" labels.
-   Creates a dataset of images and JSON label files.
-   Generates a CSV annotation file compatible with object detection training.
-   Includes a label map for object classes.

## Dependencies

-   OpenCV (`cv2`)
-   NumPy (`numpy`)
-   Random (`random`)
-   JSON (`json`)
-   sys (`sys`)

## Files

-   `Generator.py`: Contains the `Generator` and `GenerateDataset` classes.
-   `Assets/PseudoAlgo.py`: Contains algorithm templates used to generate flowcharts.
-   `dataset/`: Directory where generated images and JSON files are stored.
-   `label_map.txt`: Text file containing object class labels and IDs.
-   `annotations.csv`: CSV file containing bounding box annotations for each object in the images.

## Usage

1.  **Ensure Dependencies are Installed:**

    ```bash
    pip install opencv-python numpy
    ```

2.  **Run the script:**

    ```bash
    python Generator.py
    ```

    This will generate flowchart images and their corresponding JSON files in the `dataset/` directory.

3.  **Customize Dataset Generation:**

    -   Modify the `Range` variable in the `if __name__ == "__main__":` block to control the number of images generated.
    -   Adjust the `ALGORITHM_TEMPLATES` in `Assets/PseudoAlgo.py` to change the flowchart structures.
    -   Change the `width` and `height` parameters in the `Generator` class constructor to modify the image dimensions.
    -   Customize the appearance of the flowchart elements (colors, font, etc.) by modifying the corresponding parameters in the `Generator` class methods.

## Classes

### `Generator`

This class handles the creation of flowchart images.

-   `__init__(self, width=800, height=1000, background_color=(255, 255, 255))`: Initializes the generator with image dimensions and background color.
-   `get_text_size(self, text)`: Calculates the size of the given text.
-   `rectangle(self, text, position=None)`: Draws a rectangle with text inside.
-   `diamond(self, text, position=None)`: Draws a diamond with text inside.
-   `oval(self, text, position=None)`: Draws an oval (rounded rectangle) with text inside.
-   `parallelogram(self, text, position=None)`: Draws a parallelogram with text inside.
-   `add_arrow(self, from_element, to_element, label=None)`: Adds an arrow connecting two elements.
-   `add_decision_branches(self, decision_element, true_element, false_element)`: Adds branches from a decision diamond.
-   `create_algorithm_flowchart(self, algorithm_steps)`: Creates a flowchart from a list of algorithm steps.
-   `save_image(self, filename)`: Saves the generated image to a file.
-   `get_labeled_data(self)`: Returns the labeled data for all elements in the flowchart.

### `GenerateDataset`

This class manages the creation of the dataset.

-   `CreateDataset(self, Range)`: Generates a dataset of flowchart images and JSON files.
-   `ScriptToCSV(self, Range)`: Creates a CSV annotation file from the generated JSON files.

## Output

The script generates the following files:

-   `dataset/flowchart_XXX.jpg`: Flowchart images.
-   `dataset/flowchart_XXX.json`: JSON files containing labeled data for each image.
-   `annotations.csv`: CSV file with bounding box annotations.
-   `label_map.txt`: Text file containing object class labels.

## Example

The `ALGORITHM_TEMPLATES` in `Assets/PseudoAlgo.py` provides example algorithm steps that the generator uses to create flowcharts. You can modify these templates or add your own to generate different flowchart structures.


# Flowchart Text Extractor

This Python script extracts text from specific rectangular regions within an image using OpenCV and Tesseract OCR. It's designed to work with flowchart images where text is contained within rectangular shapes.

## Features

-   Loads an image.
-   Crops rectangular regions based on provided coordinates.
-   Preprocesses the cropped regions (grayscale, thresholding) to improve OCR accuracy.
-   Uses Tesseract OCR to extract text from the preprocessed regions.
-   Prints the extracted text to the console.

## Dependencies

-   OpenCV (`cv2`)
-   Pytesseract (`pytesseract`)
-   Tesseract OCR (installed separately)

## Installation

1.  **Install OpenCV:**

    ```bash
    pip install opencv-python
    ```

2.  **Install Pytesseract:**

    ```bash
    pip install pytesseract
    ```

3.  **Install Tesseract OCR:**

    -   **Windows:** Download and install the Tesseract executable from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki). Add the Tesseract installation directory to your system's PATH environment variable.
    -   **macOS:** Install via Homebrew: `brew install tesseract`.
    -   **Linux (Debian/Ubuntu):** `sudo apt install tesseract-ocr`.
    -   **Linux (Fedora/CentOS):** `sudo dnf install tesseract`.

4.  **Configure Pytesseract to point to your Tesseract installation:**

    -   You may need to set the `tesseract_cmd` variable in your Python script if Pytesseract cannot automatically find the Tesseract executable. For example:

        ```python
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract' # windows
        # or
        # pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract' #macOS
        # or
        # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract' # linux
        ```
        Adjust the path according to your Tesseract installation.

## Usage

1.  **Place your image file (e.g., `synthetic_flowchart.jpg`) in the same directory as the script.**
2.  **Run the script:**

    ```bash
    python your_script_name.py
    ```

    The script will load the image, extract text from the predefined rectangular regions, and print the extracted text to the console.

3.  **Modify the `shapes` list:**

    -   The `shapes` list in the `if __name__ == "__main__":` block contains the coordinates of the rectangular regions to extract text from.
    -   Each element in the list is a list of four integers: `[x, y, w, h]`, where:
        -   `x` is the x-coordinate of the top-left corner.
        -   `y` is the y-coordinate of the top-left corner.
        -   `w` is the width of the rectangle.
        -   `h` is the height of the rectangle.
    -   Change the values in this list to specify the regions you want to extract text from.

4.  **Change the Image:**

    -   Change the `img` variable to the name of your image.

## Class

### `Extractor`

This class handles the text extraction from the image.

-   `__init__(self, img)`: Initializes the extractor with the image filename and loads the image.
-   `readRectangle(self, coordinates)`: Extracts text from a rectangular region specified by the coordinates.

## Example

The script extracts text from three predefined rectangular regions in the `synthetic_flowchart.jpg` image. The extracted text is then printed to the console.

## Notes

-   The accuracy of the text extraction depends on the quality of the image and the clarity of the text.
-   Preprocessing techniques (grayscale, thresholding) are used to improve OCR accuracy. You may need to adjust the threshold value (`150` in the script) depending on the image.
-   Tesseract OCR supports multiple languages. You can specify the language using the `lang` parameter in the `pytesseract.image_to_string()` function.