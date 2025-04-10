import cv2
import numpy as np
import random
import json
from Assets.PseudoAlgo import ALGORITHM_TEMPLATES
import sys

class Generator:
    def __init__(self, width=800, height=1000, background_color=(255, 255, 255)):
        # Create a blank white image (default 800x1000 for flowcharts)
        self.img = np.full((height, width, 3), background_color, dtype=np.uint8)
        self.width = width
        self.height = height
        self.elements = []  # To keep track of all elements
        self.connections = []  # To store connections between elements
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        self.padding = 10  # Padding inside shapes

    """Calculate the size needed for text"""
    def get_text_size(self, text):
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness)
        return text_width, text_height

    """Draw a rectangle with text inside"""
    def rectangle(self, text, position=None):
        text_width, text_height = self.get_text_size(text)
        min_width = text_width + 2 * self.padding
        min_height = text_height + 2 * self.padding

        if position is None:
            x = (self.width - min_width) // 2
            y = 50 + len(self.elements) * 100
        else:
            x, y = position

        w = max(min_width, random.randint(min_width, min_width + 50))
        h = max(min_height, random.randint(min_height, min_height + 20))

        cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        text_x = x + (w - text_width) // 2
        text_y = y + (h + text_height) // 2
        cv2.putText(self.img, text, (text_x, text_y), self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        element = {
            "id": len(self.elements),
            "shape": "rectangle",
            "text": text,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "center_x": x + w // 2,
            "center_y": y + h // 2,
            "bottom_y": y + h
        }
        self.elements.append(element)
        return element

    """Draw a diamond with text inside"""
    def diamond(self, text, position=None):
        text_width, text_height = self.get_text_size(text)
        min_width = int(1.5 * (text_width + 2 * self.padding))
        min_height = int(1.5 * (text_height + 2 * self.padding))

        if position is None:
            x = (self.width - min_width) // 2
            y = 50 + len(self.elements) * 100
        else:
            x, y = position

        w = max(min_width, random.randint(min_width, min_width + 50))
        h = max(min_height, random.randint(min_height, min_height + 30))

        center_x = x + w // 2
        center_y = y + h // 2
        points = np.array([
            [center_x, y],  # top
            [x + w, center_y],  # right
            [center_x, y + h],  # bottom
            [x, center_y]  # left
        ], np.int32)

        cv2.polylines(self.img, [points], True, (0, 0, 0), 2)
        cv2.putText(self.img, text, (center_x - text_width // 2, center_y + text_height // 2),
                    self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        element = {
            "id": len(self.elements),
            "shape": "diamond",
            "text": text,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "center_x": center_x,
            "center_y": center_y,
            "bottom_y": y + h
        }
        self.elements.append(element)
        return element

    """Create a rounded rectangle for start/end instead of oval"""
    def oval(self, text, position=None):
        text_width, text_height = self.get_text_size(text)
        min_width = text_width + 2 * self.padding
        min_height = text_height + 2 * self.padding

        if position is None:
            x = (self.width - min_width) // 2
            y = 50 + len(self.elements) * 100
        else:
            x, y = position

        w = max(min_width, random.randint(min_width, min_width + 40))
        h = max(min_height, random.randint(min_height, min_height + 20))

        center_x = x + w // 2
        center_y = y + h // 2

        cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(self.img, text, (center_x - text_width // 2, center_y + text_height // 2),
                    self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        element = {
            "id": len(self.elements),
            "shape": "oval",
            "text": text,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "center_x": center_x,
            "center_y": center_y,
            "bottom_y": y + h
        }
        self.elements.append(element)
        return element

    """Draw a parallelogram (for input/output operations)"""
    def parallelogram(self, text, position=None):
        text_width, text_height = self.get_text_size(text)
        min_width = text_width + 3 * self.padding
        min_height = text_height + 2 * self.padding

        if position is None:
            x = (self.width - min_width) // 2
            y = 50 + len(self.elements) * 100
        else:
            x, y = position

        w = max(min_width, random.randint(min_width, min_width + 50))
        h = max(min_height, random.randint(min_height, min_height + 20))

        slant = int(h * 0.3)
        points = np.array([
            [x + slant, y],  # top-left
            [x + w, y],  # top-right
            [x + w - slant, y + h],  # bottom-right
            [x, y + h]  # bottom-left
        ], np.int32)

        cv2.polylines(self.img, [points], True, (0, 0, 0), 2)
        center_x = x + (w // 2)
        center_y = y + (h // 2)
        cv2.putText(self.img, text, (center_x - text_width // 2, center_y + text_height // 2),
                    self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        element = {
            "id": len(self.elements),
            "shape": "parallelogram",
            "text": text,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "center_x": center_x,
            "center_y": center_y,
            "bottom_y": y + h
        }
        self.elements.append(element)
        return element

    """Add an arrow connecting two elements"""
    def add_arrow(self, from_element, to_element, label=None):
        start_x = from_element["center_x"]
        start_y = from_element["y"] + from_element["h"]
        end_x = to_element["center_x"]
        end_y = to_element["y"]

        cv2.arrowedLine(self.img, (start_x, start_y), (end_x, end_y), (0, 0, 0), 2, tipLength=0.03)

        if label:
            mid_x = (start_x + end_x) // 2
            mid_y = (start_y + end_y) // 2
            text_width, text_height = self.get_text_size(label)
            cv2.rectangle(self.img,
                          (mid_x - text_width // 2 - 5, mid_y - text_height // 2 - 5),
                          (mid_x + text_width // 2 + 5, mid_y + text_height // 2 + 5),
                          (255, 255, 255), -1)
            cv2.putText(self.img, label, (mid_x - text_width // 2, mid_y + text_height // 2),
                        self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        connection = {
            "from_id": from_element["id"],
            "to_id": to_element["id"],
            "label": label
        }
        self.connections.append(connection)
        return connection

    """Add True/False branches from a decision diamond"""
    def add_decision_branches(self, decision_element, true_element, false_element):
        true_start_x = decision_element["x"] + decision_element["w"]
        true_start_y = decision_element["center_y"]
        true_mid_x = true_start_x + 30
        true_mid_y = true_start_y
        true_end_x = true_element["center_x"]
        true_end_y = true_element["y"]

        cv2.line(self.img, (true_start_x, true_start_y), (true_mid_x, true_mid_y), (0, 0, 0), 2)
        cv2.arrowedLine(self.img, (true_mid_x, true_mid_y), (true_end_x, true_end_y), (0, 0, 0), 2, tipLength=0.03)
        cv2.putText(self.img, "True", (true_start_x + 5, true_start_y - 5),
                    self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        false_start_x = decision_element["center_x"]
        false_start_y = decision_element["y"] + decision_element["h"]
        false_end_x = false_element["center_x"]
        false_end_y = false_element["y"]

        cv2.arrowedLine(self.img, (false_start_x, false_start_y), (false_end_x, false_end_y), (0, 0, 0), 2,
                        tipLength=0.03)
        cv2.putText(self.img, "False", (false_start_x + 5, false_start_y + 15),
                    self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        self.connections.append({
            "from_id": decision_element["id"],
            "to_id": true_element["id"],
            "label": "True"
        })
        self.connections.append({
            "from_id": decision_element["id"],
            "to_id": false_element["id"],
            "label": "False"
        })

    """Create a complete flowchart from a list of algorithm steps"""
    def create_algorithm_flowchart(self, algorithm_steps):
        y_position = 50
        x_center = self.width // 2
        step_spacing = 100

        for i, step in enumerate(algorithm_steps):
            position = (x_center, y_position)

            if step["type"] == "start" or step["type"] == "end":
                element = self.oval(step["text"], position)
            elif step["type"] == "process":
                element = self.rectangle(step["text"], position)
            elif step["type"] == "decision":
                element = self.diamond(step["text"], position)
            elif step["type"] == "input" or step["type"] == "output":
                element = self.parallelogram(step["text"], position)

            y_position = element["bottom_y"] + step_spacing
            element["step_type"] = step["type"]

            if step["type"] == "decision" and "true_branch" in step and "false_branch" in step:
                element["true_branch"] = step["true_branch"]
                element["false_branch"] = step["false_branch"]

        for i, element in enumerate(self.elements):
            if i == len(self.elements) - 1:
                continue

            if element["step_type"] == "decision":
                true_element = self.elements[element["true_branch"]]
                false_element = self.elements[element["false_branch"]]
                self.add_decision_branches(element, true_element, false_element)
            else:
                is_branch_target = False
                for e in self.elements:
                    if "true_branch" in e and (e["true_branch"] == i + 1 or e["false_branch"] == i + 1):
                        is_branch_target = True
                        break

                if not is_branch_target:
                    next_index = i + 1
                    while next_index < len(self.elements):
                        next_element = self.elements[next_index]
                        self.add_arrow(element, next_element)
                        break

        return self.elements, self.connections

    """Save the image to file"""
    def save_image(self, filename):
        cv2.imwrite(filename, self.img)
        return filename

    """Get labeled data for all elements in the flowchart"""
    def get_labeled_data(self):
        return {
            "elements": self.elements,
            "connections": self.connections
        }


class GenerateDataset():
    def CreateDataset(self, Range):
        for i in range(Range):
            generator = Generator(width=800, height=1200)
            key=random.choice(list(ALGORITHM_TEMPLATES.keys()))
            try:
                elements, connections = generator.create_algorithm_flowchart(ALGORITHM_TEMPLATES[key])
                generator.save_image(f"dataset/flowchart_{i:03d}.jpg")
                with open(f"dataset/flowchart_{i:03d}.json", "w") as f:
                    json.dump(generator.get_labeled_data(), f, indent=2)
                # bar = "|"*(i//50)
                # sys.stdout.write(f'\rCreateDataset Iteration {i//100 + 1}00: [{bar}]%')
                # sys.stdout.flush()
            except Exception as e:
                print(f"Key: {key}\nException: {e}")
                break
    def ScriptToCSV(self, Range):
        with open('label_map.txt', 'w') as f:
            f.write("item { id: 1 name: 'oval' }\n")
            f.write("item { id: 2 name: 'diamond' }\n")
            f.write("item { id: 3 name: 'rectangle' }\n")

        with open('annotations.csv', 'w') as csv_file:
            csv_file.write("filename,xmin,ymin,xmax,ymax,label\n")
            for i in range(Range):
                try:
                    with open(f"dataset/flowchart_{i:03d}.json", 'r') as f:
                        data = json.load(f)
                    for element in data["elements"]:
                        csv_file.write(
                            f"dataset/flowchart_{i:03d}.jpg,{element['x']},{element['y']},{element['x'] + element['w']},{element['y'] + element['h']},{element['shape']}\n")
                except Exception as e:
                    print(f"{i} : {e}")

if __name__ == "__main__":
    obj = GenerateDataset()
    obj.CreateDataset(20)
    obj.ScriptToCSV(20)
