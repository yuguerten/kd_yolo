import os
import xml.etree.ElementTree as ET
import shutil

def convert_to_yolo_format(xml_folder, output_folder, image_width, image_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images_train_folder = os.path.join(output_folder, "images/train")
    images_test_folder = os.path.join(output_folder, "images/test")
    labels_train_folder = os.path.join(output_folder, "labels/train")
    labels_test_folder = os.path.join(output_folder, "labels/test")

    for folder in [images_train_folder, images_test_folder, labels_train_folder, labels_test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        yolo_annotations = []
        for obj in root.findall("object"):
            label = obj.find("label").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Normalize coordinates
            center_x = ((xmin + xmax) / 2) / image_width
            center_y = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

        # Determine if the file is for training or testing
        if "train" in xml_file:
            labels_folder = labels_train_folder
            images_folder = images_train_folder
        else:
            labels_folder = labels_test_folder
            images_folder = images_test_folder

        # Save YOLO annotations to a .txt file
        output_file = os.path.join(labels_folder, os.path.splitext(xml_file)[0] + ".txt")
        with open(output_file, "w") as f:
            f.write("\n".join(yolo_annotations))

        # Copy the corresponding image to the images folder
        image_file = os.path.splitext(xml_file)[0] + ".jpg"
        shutil.copy(os.path.join(xml_folder, image_file), os.path.join(images_folder, image_file))

if __name__ == "__main__":
    xml_folder = "/home/yuguerten/workspace/kd_yolo/data/raw/archive/tuberculosis-phonecamera"
    output_folder = "/home/yuguerten/workspace/kd_yolo/data"
    image_width = 1632
    image_height = 1224

    convert_to_yolo_format(xml_folder, output_folder, image_width, image_height)
