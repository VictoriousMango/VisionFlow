import cv2
import numpy as np
import random
import json
from Assets.PseudoAlgo import ALGORITHM_TEMPLATES
import sys

import cv2
import numpy as np
import random


class Generator:
    def __init__(self, width=800, height=1000, background_color=(255, 255, 255)):
        self.img = np.full((height, width, 3), background_color, dtype=np.uint8)
        self.width = width
        self.height = height
        self.elements = []
        self.connections = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        self.padding = 10
        self.min_spacing = 50  # Minimum vertical spacing to prevent overlap

    def get_text_size(self, text):
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.font_thickness)
        return text_width, text_height

    def rectangle(self, text, position=None):
        text_width, text_height = self.get_text_size(text)
        min_width = text_width + 2 * self.padding
        min_height = text_height + 2 * self.padding

        if position is None:
            x = (self.width // (len(self.elements) + 1)) * (len(self.elements) % 4 + 1)  # Spread horizontally
            y = 50 + max((e["bottom_y"] for e in self.elements), default=0) + self.min_spacing
        else:
            x, y = position

        w = max(min_width, min_width + random.randint(0, 50))
        h = max(min_height, min_height + random.randint(0, 20))

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

    def diamond(self, text, position=None):
        text_width, text_height = self.get_text_size(text)
        min_width = int(1.5 * (text_width + 2 * self.padding))
        min_height = int(1.5 * (text_height + 2 * self.padding))

        if position is None:
            x = (self.width // (len(self.elements) + 1)) * (len(self.elements) % 4 + 1)
            y = 50 + max((e["bottom_y"] for e in self.elements), default=0) + self.min_spacing
        else:
            x, y = position

        w = max(min_width, min_width + random.randint(0, 50))
        h = max(min_height, min_height + random.randint(0, 30))

        center_x = x + w // 2
        center_y = y + h // 2
        points = np.array([
            [center_x, y],
            [x + w, center_y],
            [center_x, y + h],
            [x, center_y]
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

    def oval(self, text, position=None):
        text_width, text_height = self.get_text_size(text)
        min_width = text_width + 2 * self.padding
        min_height = text_height + 2 * self.padding

        if position is None:
            x = (self.width // (len(self.elements) + 1)) * (len(self.elements) % 4 + 1)
            y = 50 + max((e["bottom_y"] for e in self.elements), default=0) + self.min_spacing
        else:
            x, y = position

        w = max(min_width, min_width + random.randint(0, 40))
        h = max(min_height, min_height + random.randint(0, 20))

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

    def parallelogram(self, text, position=None):
        text_width, text_height = self.get_text_size(text)
        min_width = text_width + 3 * self.padding
        min_height = text_height + 2 * self.padding

        if position is None:
            x = (self.width // (len(self.elements) + 1)) * (len(self.elements) % 4 + 1)
            y = 50 + max((e["bottom_y"] for e in self.elements), default=0) + self.min_spacing
        else:
            x, y = position

        w = max(min_width, min_width + random.randint(0, 50))
        h = max(min_height, min_height + random.randint(0, 20))

        slant = int(h * 0.3)
        points = np.array([
            [x + slant, y],
            [x + w, y],
            [x + w - slant, y + h],
            [x, y + h]
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

    def add_arrow(self, from_element, to_element, label=None):
        # Start from the bottom center of from_element
        start_x = from_element["center_x"]
        start_y = from_element["bottom_y"]

        # End at the top center of to_element, avoiding unconnected shapes
        end_x = to_element["center_x"]
        end_y = to_element["y"]

        # Check for potential crossings and adjust path if needed
        min_x = min(from_element["x"], to_element["x"])
        max_x = max(from_element["x"] + from_element["w"], to_element["x"] + to_element["w"])
        for elem in self.elements:
            if elem["id"] != from_element["id"] and elem["id"] != to_element["id"]:
                if (min_x <= elem["center_x"] <= max_x and
                        min(start_y, end_y) < elem["center_y"] < max(start_y, end_y)):
                    # Shift the path to the side to avoid crossing
                    if from_element["center_x"] < self.width // 2:
                        start_x += 50
                        end_x += 50
                    else:
                        start_x -= 50
                        end_x -= 50

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

        self.connections.append({"from_id": from_element["id"], "to_id": to_element["id"], "label": label})
        return self.connections[-1]

    def add_decision_branches(self, decision_element, true_element, false_element):
        # True branch from right center to true_element
        true_start_x = decision_element["x"] + decision_element["w"]
        true_start_y = decision_element["center_y"]
        true_end_x = true_element["center_x"]
        true_end_y = true_element["y"]

        # False branch from bottom center to false_element
        false_start_x = decision_element["center_x"]
        false_start_y = decision_element["y"] + decision_element["h"]
        false_end_x = false_element["center_x"]
        false_end_y = false_element["y"]

        # Avoid crossing by checking other elements
        for elem in self.elements:
            if elem["id"] != decision_element["id"] and elem["id"] != true_element["id"]:
                if (min(true_start_x, true_end_x) <= elem["center_x"] <= max(true_start_x, true_end_x) and
                        min(true_start_y, true_end_y) < elem["center_y"] < max(true_start_y, true_end_y)):
                    true_end_x += 50 if true_end_x < self.width // 2 else -50

            if elem["id"] != decision_element["id"] and elem["id"] != false_element["id"]:
                if (min(false_start_x, false_end_x) <= elem["center_x"] <= max(false_start_x, false_end_x) and
                        min(false_start_y, false_end_y) < elem["center_y"] < max(false_start_y, false_end_y)):
                    false_end_x += 50 if false_end_x < self.width // 2 else -50

        cv2.line(self.img, (true_start_x, true_start_y), (true_end_x, true_end_y), (0, 0, 0), 2)
        cv2.arrowedLine(self.img, (true_end_x, true_end_y), (true_element["center_x"], true_element["y"]),
                        (0, 0, 0), 2, tipLength=0.03)
        cv2.putText(self.img, "True", (true_start_x + 5, true_start_y - 5),
                    self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        cv2.arrowedLine(self.img, (false_start_x, false_start_y), (false_end_x, false_end_y),
                        (0, 0, 0), 2, tipLength=0.03)
        cv2.arrowedLine(self.img, (false_end_x, false_end_y), (false_element["center_x"], false_element["y"]),
                        (0, 0, 0), 2, tipLength=0.03)
        cv2.putText(self.img, "False", (false_start_x + 5, false_start_y + 15),
                    self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        self.connections.append({"from_id": decision_element["id"], "to_id": true_element["id"], "label": "True"})
        self.connections.append({"from_id": decision_element["id"], "to_id": false_element["id"], "label": "False"})

    def create_algorithm_flowchart(self, algorithm_steps):
        y_position = 50
        x_positions = [self.width // 5, 2 * self.width // 5, 3 * self.width // 5, 4 * self.width // 5]  # 4 columns

        for i, step in enumerate(algorithm_steps):
            # Assign x based on column to spread elements
            x = x_positions[i % len(x_positions)]
            position = (x, y_position)

            if step["type"] == "start" or step["type"] == "end":
                element = self.oval(step["text"], position)
            elif step["type"] == "process":
                element = self.rectangle(step["text"], position)
            elif step["type"] == "decision":
                element = self.diamond(step["text"], position)
            elif step["type"] == "input" or step["type"] == "output":
                element = self.parallelogram(step["text"], position)

            y_position = max(y_position, element["bottom_y"]) + self.min_spacing  # Adjust y to prevent overlap
            element["step_type"] = step["type"]

            if step["type"] == "decision" and "true_branch" in step and "false_branch" in step:
                element["true_branch"] = step["true_branch"]
                element["false_branch"] = step["false_branch"]

        # Connect elements based on sequence and decisions
        for i, element in enumerate(self.elements):
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
                if not is_branch_target and i < len(self.elements) - 1:
                    next_element = self.elements[i + 1]
                    self.add_arrow(element, next_element)

        return self.elements, self.connections

    def save_image(self, filename):
        cv2.imwrite(filename, self.img)
        return filename

    def get_labeled_data(self):
        return {"elements": self.elements, "connections": self.connections}


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
