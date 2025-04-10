import os
import json
import subprocess

class J2C():
    def __int__(self):
        pass
    def convert_to_coco(self, json_dir, output_file, image_dir):
        # Initialize COCO structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Define categories (match your shape types)
        categories = {
            "oval": 1,
            "diamond": 2,
            "rectangle": 3
        }
        for name, id in categories.items():
            coco_data["categories"].append({"id": id, "name": name})

        # Assign unique IDs
        image_id = 0
        annotation_id = 0

        # Process each JSON file
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        json_files.sort()

        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Add image metadata
            image_file = json_file.replace('.json', '.jpg')
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": 800,  # Match your Generator width
                "height": 1000  # Match your Generator height
            })

            # Add annotations for each element
            for element in data["elements"]:
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": categories[element["shape"]],
                    "bbox": [element["x"], element["y"], element["w"], element["h"]],  # [x_min, y_min, width, height]
                    "area": element["w"] * element["h"],  # Calculate area
                    "iscrowd": 0,
                    "text": element.get("text", "")  # Optional, for your pipeline
                })
                annotation_id += 1

            image_id += 1

        # Save to COCO JSON
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"COCO JSON saved to {output_file}")

    def TrainValSplit(self):
        with open("dataset/instances_coco.json", 'r') as f:
            data = json.load(f)

        split_index = int(len(data["images"]) * 0.8)
        train_data = {
            "images": data["images"][:split_index],
            "annotations": [ann for ann in data["annotations"] if ann["image_id"] < split_index],
            "categories": data["categories"]
        }
        val_data = {
            "images": data["images"][split_index:],
            "annotations": [ann for ann in data["annotations"] if ann["image_id"] >= split_index],
            "categories": data["categories"]
        }

        with open("dataset/instances_train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        with open("dataset/instances_val.json", 'w') as f:
            json.dump(val_data, f, indent=2)

    def create_coco_tf_record(self, image_dir, annotations_file, output_file_prefix, num_shards):
        """
        Executes the create_coco_tf_record.py command using the OS library.

        Args:
            image_dir (str): The path to the image directory.
            annotations_file (str): The path to the object annotations JSON file.
            output_file_prefix (str): The prefix for the output TFRecord files.
            num_shards (int): The number of output shards.
        """

        command = [
            "python",
            "E:/Programs/Temp Working Area/models/research/object_detection/dataset_tools/create_coco_tf_record.py",
            "--logtostderr",
            f"--image_dir={image_dir}",
            f"--object_annotations_file={annotations_file}",
            f"--output_file_prefix={output_file_prefix}",
            f"--num_shards={num_shards}",
        ]

        try:
            # Using subprocess.run is generally preferred over os.system for better control
            # and error handling.
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Command executed successfully.")
            print("Standard Output:", result.stdout)
            print("Standard Error:", result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            print("Standard Output:", e.stdout)
            print("Standard Error:", e.stderr)
        except FileNotFoundError:
            print("Error: python or the create_coco_tf_record.py script was not found. Please verify your environment and paths.")
        except Exception as generic_exception:
            print(f"An unexpected error occurred: {generic_exception}")



if __name__ == "__main__":
    json_dir = "dataset"  # Directory with your JSON files
    output_file = "dataset/instances_coco.json"  # Output COCO file
    image_dir = "dataset"  # Directory with your images
    obj = J2C()
    # obj.convert_to_coco(json_dir, output_file, image_dir)
    # Example usage:
    image_dir = "dataset"
    annotations_file = "dataset/instances_train.json"
    output_file_prefix = "tfrecords/train"
    num_shards = 1

    obj.create_coco_tf_record(image_dir, annotations_file, output_file_prefix, num_shards)