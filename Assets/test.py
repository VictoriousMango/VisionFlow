import re
import cv2
import numpy as np
import pytesseract
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import io


# Image preprocessing functions
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def get_opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):
    return cv2.Canny(image, 100, 200)


# Detect shapes in the flowchart
def detect_shapes(image, thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)

        # Filter small contours (noise)
        if area < 100:
            continue

        # Classify shapes
        if len(approx) >= 8:  # Oval (Start/End)
            shape_type = "oval"
        elif len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2:  # Diamond (Decision)
            shape_type = "diamond"
        elif len(approx) == 4:  # Rectangle (Process)
            shape_type = "rectangle"
        else:
            continue

        shapes.append({
            'type': shape_type,
            'contour': approx,
            'bbox': (x, y, w, h),
            'center': (x + w // 2, y + h // 2)
        })
    return shapes


# Extract text from a region
def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]
    config = r'--oem 3 --psm 6'
    try:
        text = pytesseract.image_to_string(roi, config=config).strip()
        return text
    except Exception as e:
        return ""


# Detect arrows using Hough Line Transform
def detect_arrows(canny, shapes):
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
    edges = []
    if lines is None:
        return edges

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Find nearest shapes
        start_shape, end_shape = None, None
        min_start_dist = float('inf')
        min_end_dist = float('inf')

        for i, shape in enumerate(shapes):
            cx, cy = shape['center']
            dist_start = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5
            dist_end = ((x2 - cx) ** 2 + (y2 - cy) ** 2) ** 0.5
            if dist_start < min_start_dist:
                min_start_dist = dist_start
                start_shape = i
            if dist_end < min_end_dist:
                min_end_dist = dist_end
                end_shape = i

        if start_shape is not None and end_shape is not None and start_shape != end_shape:
            edges.append((start_shape, end_shape))

    return edges


# Parse text to identify algorithmic constructs
def parse_algorithm_text(text):
    text = text.lower()
    if re.search(r'\b(start|begin)\b', text):
        return "start"
    elif re.search(r'\b(if|condition|whether)\b', text):
        return "condition"
    elif re.search(r'\b(while|for|loop)\b', text):
        return "loop"
    elif re.search(r'\b(end|stop|finish)\b', text):
        return "end"
    else:
        return "process"


# Generate pseudocode from graph and shapes
def generate_pseudocode(G, shapes):
    pseudocode = []
    visited = set()

    def dfs(node, indent=0):
        if node in visited:
            return
        visited.add(node)

        shape = shapes[node]
        text = shape.get('text', '')
        node_type = shape.get('node_type', 'process')

        indent_str = "  " * indent
        if node_type == "start":
            pseudocode.append(f"{indent_str}START")
        elif node_type == "end":
            pseudocode.append(f"{indent_str}END")
        elif node_type == "condition":
            pseudocode.append(f"{indent_str}IF {text}")
            # Handle true/false branches
            successors = list(G.successors(node))
            for i, succ in enumerate(successors):
                branch = "THEN" if i == 0 else "ELSE"
                pseudocode.append(f"{indent_str}  {branch}")
                dfs(succ, indent + 2)
        elif node_type == "loop":
            pseudocode.append(f"{indent_str}WHILE {text}")
            for succ in G.successors(node):
                dfs(succ, indent + 1)
        else:
            pseudocode.append(f"{indent_str}PROCESS: {text}")
            for succ in G.successors(node):
                dfs(succ, indent)

    # Find start node
    start_node = next((i for i, s in enumerate(shapes) if s['node_type'] == "start"), 0)
    dfs(start_node)
    return "\n".join(pseudocode)


# Main function
def main(image_path):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    gray = get_grayscale(image)
    thresh = thresholding(gray)
    canny_img = canny(gray)

    # Detect shapes
    shapes = detect_shapes(image, thresh)

    # Extract text and classify nodes
    for shape in shapes:
        shape['text'] = extract_text(image, shape['bbox'])
        shape['node_type'] = parse_algorithm_text(shape['text'])

    # Detect arrows
    edges = detect_arrows(canny_img, shapes)

    # Build graph
    G = nx.DiGraph()
    G.add_nodes_from(range(len(shapes)))
    G.add_edges_from(edges)

    # Generate pseudocode
    pseudocode = generate_pseudocode(G, shapes)

    # Visualize results
    plt.figure(figsize=(10, 8))
    for shape in shapes:
        x, y, w, h = shape['bbox']
        if shape['type'] == "rectangle":
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue')
            plt.gca().add_patch(rect)
        elif shape['type'] == "diamond":
            points = [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)]
            diamond = plt.Polygon(points, fill=False, edgecolor='red')
            plt.gca().add_patch(diamond)
        elif shape['type'] == "oval":
            ellipse = plt.Ellipse((x + w / 2, y + h / 2), w, h, fill=False, edgecolor='green')
            plt.gca().add_patch(ellipse)
        plt.text(x, y, shape['text'], fontsize=8)

    for edge in edges:
        start = shapes[edge[0]]['center']
        end = shapes[edge[1]]['center']
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  head_width=10, head_length=10, fc='black', ec='black')

    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Detected Flowchart")
    plt.show()

    # Print pseudocode
    print("Extracted Algorithm (Pseudocode):")
    print("---------------------------------")
    print(pseudocode)


# Run the script
if __name__ == "__main__":
    # Replace with your image path in Colab (e.g., '/content/workflow_manager.png')
    image_path = "./img.png"
    main(image_path)