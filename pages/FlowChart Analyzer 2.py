import streamlit as st
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import uuid
import re
import requests
import os
from dotenv import load_dotenv
from skimage import measure

# Load environment variables
load_dotenv()
tesseract_path = os.getenv("TESSERACT_PATH")
groq_api_key = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Set Tesseract path if provided
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


class ImagePreprocessor:
    """Class to handle image preprocessing steps for workflow diagrams."""

    @staticmethod
    def get_grayscale(image):
        """Convert image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def remove_noise(image):
        """Remove noise using median blur."""
        return cv2.medianBlur(image, 5)

    @staticmethod
    def thresholding(image):
        """Apply OTSU thresholding."""
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    @staticmethod
    def adaptive_threshold(image):
        """Apply adaptive thresholding."""
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

    @staticmethod
    def dilate(image):
        """Dilate the image to fill gaps."""
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    @staticmethod
    def erode(image):
        """Erode the image to remove small noise."""
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    @staticmethod
    def get_opening(image):
        """Apply morphological opening."""
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def get_closing(image):
        """Apply morphological closing."""
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def canny(image):
        """Apply Canny edge detection."""
        return cv2.Canny(image, 100, 200)

    def preprocess(self, image):
        """Apply complete preprocessing pipeline and return results."""
        # Convert to grayscale
        gray = self.get_grayscale(image)

        # Apply thresholding
        thresh = self.thresholding(gray)
        adaptive = self.adaptive_threshold(gray)

        # Apply morphological operations
        opening = self.get_opening(gray)
        closing = self.get_closing(gray)

        # Apply edge detection
        canny = self.canny(gray)

        # Create binary image for shape detection
        binary = adaptive.copy()
        binary = self.get_closing(binary)

        return {
            'grayscale': gray,
            'thresh': thresh,
            'adaptive': adaptive,
            'opening': opening,
            'closing': closing,
            'canny': canny,
            'binary': binary,
            'original': image
        }


class OCRProcessor:
    """Class to handle OCR processing with Pytesseract."""

    def __init__(self, config=r'--oem 3 --psm 6'):
        """Initialize with OCR configuration."""
        self.config = config

    def extract_text(self, image):
        """Extract text from an image using Pytesseract."""
        return pytesseract.image_to_string(image, config=self.config)

    def extract_text_from_roi(self, image, bbox):
        """Extract text from a region of interest."""
        x, y, w, h = bbox
        roi = image[y:y + h, x:x + w]

        # Convert ROI to grayscale if needed
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi

        # Preprocess ROI for better OCR
        _, roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Use pytesseract to extract text
        text = pytesseract.image_to_string(roi_thresh, config=self.config)

        # Clean the text
        text = text.strip()

        return text

    def process_images(self, images):
        """Extract text from multiple preprocessed images."""
        extracted_texts = {}
        for key, img in images.items():
            extracted_texts[key] = self.extract_text(img)
        return extracted_texts


class ShapeDetector:
    """Class to detect shapes in workflow diagrams."""

    def __init__(self):
        """Initialize shape detector."""
        self.min_area = 500  # Minimum area threshold to filter out noise

    def detect_shapes(self, binary_image, original_image, ocr_processor):
        """Detect shapes in the workflow diagram."""
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours (noise)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]

        shapes = []
        for cnt in filtered_contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Get shape attributes
            x, y, w, h = cv2.boundingRect(cnt)
            shape_type = self.classify_shape(approx)

            # Extract text inside shape using OCR
            text = ocr_processor.extract_text_from_roi(original_image, (x, y, w, h))

            shapes.append({
                'type': shape_type,
                'contour': cnt,
                'bbox': (x, y, w, h),
                'text': text
            })

        return shapes

    def classify_shape(self, approx):
        """Classify shape based on number of vertices."""
        vertices = len(approx)

        if vertices == 4:
            # Check if it's a rectangle or square
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            if 0.9 <= aspect_ratio <= 1.1:
                return "square"  # Could be a process/action
            else:
                return "rectangle"  # Could be a process/entity/table

        elif vertices == 3:
            return "triangle"  # Could be a data flow or connector

        elif vertices == 5:
            return "pentagon"  # Could be a document

        elif vertices == 6:
            return "hexagon"  # Could be a preparation step

        elif vertices > 6:
            # Might be a circle or oval
            return "circle"  # Could be a start/end point or decision

        else:
            return "unknown"


class ConnectionDetector:
    """Class to detect connections between shapes in workflow diagrams."""

    def detect_connections(self, binary_image, shapes):
        """Detect connections (arrows, lines) between shapes."""
        # Get a copy of the binary image to work with
        connection_img = binary_image.copy()

        # Create a mask where all shapes are filled in white
        shape_mask = np.zeros_like(binary_image)
        for shape in shapes:
            cv2.drawContours(shape_mask, [shape['contour']], 0, 255, -1)

        # Subtract shapes from binary image to keep only connections
        connection_img = cv2.bitwise_and(connection_img, cv2.bitwise_not(shape_mask))

        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(connection_img, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=20)

        connections = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Find which shapes this line connects
                start_shape = self.find_closest_shape(shapes, (x1, y1))
                end_shape = self.find_closest_shape(shapes, (x2, y2))

                if start_shape is not None and end_shape is not None and start_shape != end_shape:
                    connections.append({
                        'start': start_shape,
                        'end': end_shape,
                        'line': (x1, y1, x2, y2)
                    })

        return connections

    def find_closest_shape(self, shapes, point):
        """Find the shape closest to a given point."""
        closest_shape = None
        min_distance = float('inf')

        x, y = point

        for i, shape in enumerate(shapes):
            shape_x, shape_y, shape_w, shape_h = shape['bbox']

            # Check if point is inside the shape
            if shape_x <= x <= shape_x + shape_w and shape_y <= y <= shape_y + shape_h:
                return i

            # Calculate distance to shape center
            center_x = shape_x + shape_w / 2
            center_y = shape_y + shape_h / 2

            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_shape = i

        # Only return if the point is reasonably close to the shape
        if min_distance < 50:  # Threshold distance
            return closest_shape

        return None


class VisualizationUtils:
    """Class for visualization utilities."""

    @staticmethod
    def visualize_detection(original_image, shapes, connections):
        """Visualize detected shapes and connections."""
        # Create a copy of the original image to draw on
        visualization = original_image.copy()

        # Draw shapes
        for i, shape in enumerate(shapes):
            # Draw contour
            cv2.drawContours(visualization, [shape['contour']], 0, (0, 255, 0), 2)

            # Draw bounding box
            x, y, w, h = shape['bbox']
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Add shape type and index
            label = f"{i}: {shape['type']}"
            cv2.putText(visualization, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw connections
        for conn in connections:
            x1, y1, x2, y2 = conn['line']
            cv2.line(visualization, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Draw arrow head
            angle = np.arctan2(y2 - y1, x2 - x1)
            x_arr = int(x2 - 10 * np.cos(angle - np.pi / 6))
            y_arr = int(y2 - 10 * np.sin(angle - np.pi / 6))
            cv2.line(visualization, (x2, y2), (x_arr, y_arr), (0, 0, 255), 2)

            x_arr = int(x2 - 10 * np.cos(angle + np.pi / 6))
            y_arr = int(y2 - 10 * np.sin(angle + np.pi / 6))
            cv2.line(visualization, (x2, y2), (x_arr, y_arr), (0, 0, 255), 2)

        return visualization

    @staticmethod
    def matplotlib_to_pil(fig):
        """Convert Matplotlib figure to PIL Image."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(buf)


class WorkflowClassifier:
    """Class for classifying workflow diagrams."""

    def classify_workflow(self, shapes, connections):
        """Classify the workflow as either a database schema or procedural logic."""
        # Count shape types for classification
        shape_types = [shape['type'] for shape in shapes]

        # Heuristics for classification
        rectangle_count = shape_types.count('rectangle')
        circle_count = shape_types.count('circle')
        diamond_count = shape_types.count('diamond')

        # ER diagrams typically have more rectangles (entities) with fewer diamonds
        # Flowcharts typically have more diamonds (decisions) and circles (start/end)

        # Calculate the proportion of rectangles to total shapes
        rectangle_ratio = rectangle_count / len(shapes) if shapes else 0

        # Check for ER diagram keywords in text
        er_keywords = ['entity', 'table', 'primary key', 'foreign key', 'id', 'pk', 'fk', 'relation']
        er_keyword_count = 0

        # Check for procedural keywords in text
        proc_keywords = ['start', 'end', 'if', 'else', 'then', 'loop', 'while', 'for', 'process', 'decision']
        proc_keyword_count = 0

        for shape in shapes:
            text = shape['text'].lower()
            for keyword in er_keywords:
                if keyword in text:
                    er_keyword_count += 1

            for keyword in proc_keywords:
                if keyword in text:
                    proc_keyword_count += 1

        # Classification logic
        if (rectangle_ratio > 0.6 or er_keyword_count > proc_keyword_count) and diamond_count <= 1:
            return "database_schema", 0.8 + min(er_keyword_count * 0.05, 0.15)
        else:
            return "procedural_logic", 0.8 + min(proc_keyword_count * 0.05, 0.15)


class SQLGenerator:
    """Class for generating SQL from database schema diagrams."""

    def generate_sql_schema(self, shapes, connections):
        """Generate SQL DDL statements from an ER diagram."""
        sql_output = "-- SQL Schema Generated by VisionFlow\n\n"

        # Track tables and their relationships
        tables = {}
        relationships = []

        # First pass: identify entities/tables
        for i, shape in enumerate(shapes):
            if shape['type'] in ['rectangle', 'square']:
                # Extract table name and potential columns from text
                text = shape['text']
                lines = text.strip().split('\n')

                if not lines:
                    table_name = f"Table_{i}"
                    columns = []
                else:
                    table_name = lines[0].strip().replace(' ', '_')
                    columns = []

                    # If there are additional lines, they might be columns
                    if len(lines) > 1:
                        for line in lines[1:]:
                            line = line.strip()
                            if line:
                                columns.append(line)

                # Store table info
                tables[i] = {
                    'name': table_name,
                    'raw_columns': columns,
                    'processed_columns': []
                }

        # Second pass: process columns and extract data types
        for table_idx, table_info in tables.items():
            for col in table_info['raw_columns']:
                # Look for column name and type patterns
                col = col.strip()

                # Check for primary key indicator
                is_pk = False
                if 'pk' in col.lower() or 'primary key' in col.lower() or '#' in col:
                    is_pk = True
                    col = col.replace('PK', '').replace('pk', '').replace('primary key', '').replace('#', '').strip()

                # Try to extract column name and type
                parts = col.split()
                if len(parts) >= 1:
                    col_name = parts[0].strip().replace(',', '')

                    # Determine column type (default to VARCHAR if not found)
                    col_type = "VARCHAR(255)"
                    if len(parts) > 1:
                        type_hint = parts[1].lower()
                        if 'int' in type_hint:
                            col_type = "INTEGER"
                        elif 'char' in type_hint or 'text' in type_hint or 'string' in type_hint:
                            col_type = "VARCHAR(255)"
                        elif 'date' in type_hint:
                            col_type = "DATE"
                        elif 'time' in type_hint:
                            col_type = "TIMESTAMP"
                        elif 'float' in type_hint or 'double' in type_hint or 'decimal' in type_hint:
                            col_type = "DECIMAL(10,2)"
                        elif 'bool' in type_hint:
                            col_type = "BOOLEAN"

                    # Add primary key constraint if detected
                    constraints = "PRIMARY KEY" if is_pk else ""

                    table_info['processed_columns'].append({
                        'name': col_name,
                        'type': col_type,
                        'constraints': constraints
                    })

            # Ensure every table has at least one column (id column as default)
            if not table_info['processed_columns']:
                table_info['processed_columns'].append({
                    'name': 'id',
                    'type': 'INTEGER',
                    'constraints': 'PRIMARY KEY'
                })

        # Third pass: identify relationships from connections
        for conn in connections:
            start_idx = conn['start']
            end_idx = conn['end']

            if start_idx in tables and end_idx in tables:
                relationships.append({
                    'from_table': tables[start_idx]['name'],
                    'to_table': tables[end_idx]['name'],
                    'from_idx': start_idx,
                    'to_idx': end_idx
                })

        # Generate CREATE TABLE statements
        for table_idx, table_info in tables.items():
            sql_output += f"CREATE TABLE {table_info['name']} (\n"

            # Add columns
            column_defs = []
            for col in table_info['processed_columns']:
                col_def = f"    {col['name']} {col['type']}"
                if col['constraints']:
                    col_def += f" {col['constraints']}"
                column_defs.append(col_def)

            # Add foreign key constraints based on relationships
            for rel in relationships:
                if rel['to_idx'] == table_idx:
                    from_table = tables[rel['from_idx']]['name']
                    # Look for a primary key in the from_table
                    from_pk = None
                    for col in tables[rel['from_idx']]['processed_columns']:
                        if "PRIMARY KEY" in col['constraints']:
                            from_pk = col['name']
                            break

                    if not from_pk:
                        from_pk = "id"  # Default if no PK found

                    fk_name = f"{from_table.lower()}_id"
                    column_defs.append(f"    {fk_name} INTEGER REFERENCES {from_table}({from_pk})")

            sql_output += ",\n".join(column_defs)
            sql_output += "\n);\n\n"

        return sql_output


class CodeGenerator:
    """Class for generating Python code from procedural workflow diagrams."""

    def generate_python_code(self, shapes, connections):
        """Generate Python code from a procedural workflow diagram."""
        # Sort shapes by y-coordinate to follow top-to-bottom flow
        sorted_shapes = sorted(enumerate(shapes), key=lambda x: x[1]['bbox'][1])
        shape_order = [idx for idx, _ in sorted_shapes]

        # Map original indices to new ordering
        index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(shape_order)}

        # Remap connections based on new indices
        remapped_connections = []
        for conn in connections:
            remapped_connections.append({
                'start': index_map[conn['start']],
                'end': index_map[conn['end']],
                'line': conn['line']
            })

        # Reorder shapes
        reordered_shapes = [shapes[idx] for idx in shape_order]

        # Identify start and end nodes
        start_idx = None
        end_idx = None

        for i, shape in enumerate(reordered_shapes):
            text = shape['text'].lower()
            if 'start' in text or 'begin' in text:
                start_idx = i
            elif 'end' in text or 'stop' in text:
                end_idx = i

        # If start not found, use the topmost shape
        if start_idx is None:
            start_idx = 0

        # Build a graph representation
        graph = [[] for _ in range(len(reordered_shapes))]
        for conn in remapped_connections:
            graph[conn['start']].append(conn['end'])

        # Generate code
        code_output = "# Python code generated by VisionFlow\n\n"
        code_output += "def process_workflow():\n"

        # Process graph and generate code
        visited = set()
        indentation = 1

        def generate_block(node_idx, indent):
            nonlocal code_output, visited

            if node_idx in visited:
                return

            visited.add(node_idx)
            current_shape = reordered_shapes[node_idx]
            shape_type = current_shape['type']
            text = current_shape['text'].strip()

            # Skip explicit start/end nodes when generating code
            if ('start' in text.lower() or 'begin' in text.lower() or
                    'end' in text.lower() or 'stop' in text.lower() or 'exit' in text.lower()):
                # Just follow the outgoing connections
                for next_node in graph[node_idx]:
                    generate_block(next_node, indent)
                return

            spaces = "    " * indent

            # Handle shape based on type and text
            if shape_type == 'rectangle' or shape_type == 'square':
                # Process/action block
                # Clean up the text and make it Python-friendly
                action_text = text.replace('\n', ' ').strip()

                # Check if it's an assignment operation
                if '=' in action_text and not '==' in action_text:
                    code_output += f"{spaces}{action_text}\n"
                else:
                    # Try to make it a valid Python statement
                    # Remove any trailing punctuation
                    action_text = re.sub(r'[,.;:!?]$', '', action_text)

                    # Check for common action verbs and format accordingly
                    if re.match(r'^(print|display|show|output)', action_text, re.I):
                        # It's a print statement
                        match = re.match(r'^(print|display|show|output)\s+(.*)', action_text, re.I)
                        if match:
                            content = match.group(2)
                            code_output += f"{spaces}print({content!r})\n"
                        else:
                            code_output += f"{spaces}print({action_text!r})\n"
                    elif re.match(r'^(calculate|compute|find)', action_text, re.I):
                        # It's a calculation
                        match = re.match(r'^(calculate|compute|find)\s+(.*)', action_text, re.I)
                        if match:
                            content = match.group(2)
                            # Create a variable name from the content
                            var_name = re.sub(r'[^a-zA-Z0-9_]', '_', content.lower())
                            var_name = re.sub(r'_+', '_', var_name).strip('_')
                            if not var_name:
                                var_name = "result"
                            code_output += f"{spaces}{var_name} = {content}  # Calculate {content}\n"
                        else:
                            code_output += f"{spaces}# {action_text}\n"
                    elif re.match(r'^(set|initialize|assign)', action_text, re.I):
                        # It's an assignment
                        match = re.match(r'^(set|initialize|assign)\s+(\w+)(?:\s+to\s+|\s*=\s*)(.+)', action_text, re.I)
                        if match:
                            var_name = match.group(2)
                            value = match.group(3)
                            code_output += f"{spaces}{var_name} = {value}\n"
                        else:
                            code_output += f"{spaces}# {action_text}\n"
                    else:
                        # Generic comment
                        code_output += f"{spaces}# {action_text}\n"

            elif shape_type == 'diamond':
                # Decision block
                condition_text = text.replace('\n', ' ').strip()

                # Clean up the condition text
                condition_text = re.sub(r'[,.;:!?]$', '', condition_text)

                # Format as Python if condition
                if not condition_text:
                    condition_text = "condition"

                # Format the condition for Python syntax
                if '>' in condition_text or '<' in condition_text or '==' in condition_text:
                    # Already contains comparison operators
                    code_output += f"{spaces}if {condition_text}:\n"
                else:
                    # Add appropriate comparison
                    code_output += f"{spaces}if {condition_text}:\n"

                # Find the 'true' path (typically right or down)
                true_path = None
                false_path = None

                # Get the center of the current shape
                cx, cy = (current_shape['bbox'][0] + current_shape['bbox'][2] // 2,
                          current_shape['bbox'][1] + current_shape['bbox'][3] // 2)

                for next_node in graph[node_idx]:
                    next_shape = reordered_shapes[next_node]
                    nx, ny = (next_shape['bbox'][0] + next_shape['bbox'][2] // 2,
                              next_shape['bbox'][1] + next_shape['bbox'][3] // 2)

                    # Determine if it's a 'yes' or 'no' path based on position
                    # Typically, 'yes' is to the right or down, 'no' is to the left or up
                    if nx > cx or ny > cy:  # To the right or down
                        true_path = next_node
                    else:  # To the left or up
                        false_path = next_node

                # If we couldn't determine based on position, just use the first as true
                if true_path is None and len(graph[node_idx]) > 0:
                    true_path = graph[node_idx][0]
                    if len(graph[node_idx]) > 1:
                        false_path = graph[node_idx][1]

                # Generate the 'true' branch
                if true_path is not None:
                    generate_block(true_path, indent + 1)

                # Generate the 'false' branch if it exists
                if false_path is not None:
                    code_output += f"{spaces}else:\n"
                    generate_block(false_path, indent + 1)

                return  # Don't continue with default next nodes

            elif shape_type == 'circle':
                # Start/end or connector
                if 'start' in text.lower() or 'begin' in text.lower():
                    code_output += f"{spaces}# Start of process\n"
                elif 'end' in text.lower() or 'stop' in text.lower():
                    code_output += f"{spaces}return  # End of process\n"
                else:
                    code_output += f"{spaces}# {text}\n"

            else:
                # Generic block
                code_output += f"{spaces}# {text}\n"

            # Continue with next nodes (unless it's a decision which handles this separately)
            if shape_type != 'diamond':
                for next_node in graph[node_idx]:
                    generate_block(next_node, indent)

        # Start generating from the start node
        generate_block(start_idx, indentation)

        # Add a main block
        code_output += "\n# Execute the workflow\n"
        code_output += "if __name__ == '__main__':\n"
        code_output += "    process_workflow()\n"

        return code_output


class AIAnalyzer:
    """Class for AI-powered workflow analysis."""

    def analyze_workflow_with_groq(self, text, shapes, connections):
        """Analyze workflow text using Groq API to generate enhanced code."""
        if not groq_api_key:
            return "Error: GROQ_API_KEY not found in .env file."

        # Create structured representation of shapes and connections for the AI
        shape_data = []
        for i, shape in enumerate(shapes):
            shape_data.append({
                'id': i,
                'type': shape['type'],
                'text': shape['text'],
            })

        connection_data = []
        for conn in connections:
            connection_data.append({
                'from': conn['start'],
                'to': conn['end']
            })

        # Create prompt with structured representation
        prompt = f"""
        I have a workflow diagram with the following components:

        EXTRACTED TEXT:
        {text}

        SHAPES:
        {shape_data}

        CONNECTIONS:
        {connection_data}

        Please analyze this workflow and generate clean, efficient Python code that implements it.
        Add proper error handling, comments, and ensure the control flow logic is preserved.
        """

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system",
                 "content": "You are an expert in analyzing workflow diagrams and generating python code. Extract the sequence of steps, decisions, and actions from the provided text and output structured python code with comments."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
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

def main():
    """Main application function for Streamlit UI."""
    st.set_page_config(page_title="VisionFlow", layout="wide")
    st.title("VisionFlow - Workflow Diagram Analyzer")
    st.write("Upload an image of a workflow diagram to generate SQL schema or Python code.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a workflow diagram image...",
                                     type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Convert BGR to RGB for display
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Initialize processors
        preprocessor = ImagePreprocessor()
        ocr_processor = OCRProcessor()
        shape_detector = ShapeDetector()
        connection_detector = ConnectionDetector()
        workflow_classifier = WorkflowClassifier()
        sql_generator = SQLGenerator()
        code_generator = CodeGenerator()
        ai_analyzer = AIAnalyzer()

        # Display original image
        st.subheader("Input Workflow Diagram")
        st.image(rgb_image, use_column_width=True)

        # Preprocess image
        with st.spinner("Processing image..."):
            preprocessed = preprocessor.preprocess(original_image)
            binary_image = preprocessed['binary']

            # Detect shapes
            shapes = shape_detector.detect_shapes(binary_image, original_image, ocr_processor)

            # Detect connections
            connections = connection_detector.detect_connections(binary_image, shapes)

            # Visualize detection
            visualization = VisualizationUtils.visualize_detection(
                rgb_image.copy(), shapes, connections)

            # Classify workflow
            workflow_type, confidence = workflow_classifier.classify_workflow(shapes, connections)

            # Generate output based on workflow type
            if workflow_type == "database_schema":
                output = sql_generator.generate_sql_schema(shapes, connections)
                output_type = "SQL Schema"
            else:
                output = code_generator.generate_python_code(shapes, connections)
                output_type = "Python Code"

            # Get extracted text for AI analysis
            extracted_text = "\n".join([shape['text'] for shape in shapes])

        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Detection Results",
            output_type,
            "AI Enhanced Analysis",
            "Extracted Text"
        ])

        with tab1:
            st.subheader("Detected Shapes and Connections")
            st.image(visualization, use_column_width=True)

            st.write(f"**Workflow Type:** {workflow_type.replace('_', ' ').title()} "
                     f"(confidence: {confidence:.0%})")

            st.subheader("Detected Shapes")
            for i, shape in enumerate(shapes):
                st.write(f"**Shape {i}** ({shape['type']}): {shape['text']}")

        with tab2:
            st.subheader(f"Generated {output_type}")
            st.code(output, language='sql' if workflow_type == "database_schema" else 'python')

            # Download button
            file_ext = "sql" if workflow_type == "database_schema" else "py"
            st.download_button(
                label=f"Download {output_type}",
                data=output,
                file_name=f"workflow_output.{file_ext}",
                mime="text/plain"
            )

        with tab3:
            st.subheader("AI Enhanced Analysis")
            with st.spinner("Analyzing with AI..."):
                ai_output = ai_analyzer.analyze_workflow_with_groq(
                    extracted_text, shapes, connections)

            st.code(ai_output, language='python')

            st.download_button(
                label="Download AI Enhanced Code",
                data=ai_output,
                file_name="ai_enhanced_code.py",
                mime="text/plain"
            )

        with tab4:
            st.subheader("Extracted Text from Workflow")
            st.text_area("OCR Results", extracted_text, height=300)

if __name__ == "__main__":
    main()