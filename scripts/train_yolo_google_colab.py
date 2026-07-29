"""
Script de treino YOLOv8 no Google Colab.

Dataset: carlosrian/software-architecture-dataset (Kaggle)
Resultados: Google Drive → yolov8_training_results/software_architecture_model/

Após o treino, copie weights/best.pt para:
  models/software_architecture_model/weights/best.pt
"""

# !pip install ultralytics kagglehub

import os
import xml.etree.ElementTree as ET
import random
import shutil

import kagglehub
from ultralytics import YOLO
from google.colab import drive

drive.mount("/content/drive")

print(50 * "=")
print("Baixando o dataset do kaggle...")
raw_data_path = kagglehub.dataset_download("carlosrian/software-architecture-dataset")
print(50 * "=")
print("Caminho para os arquivos do dataset:", raw_data_path)
print(50 * "=")

YOLO_OUTPUT_DIR = "/content/yolo_dataset"

print("Mapeando classes dos arquivos XML...")
print(50 * "=")
all_xml_files = []
for root_dir, _, files in os.walk(raw_data_path):
    for file in files:
        if file.lower().endswith(".xml"):
            all_xml_files.append(os.path.join(root_dir, file))

detected_classes = set()
for xml_file in all_xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.iter("object"):
            name_node = obj.find("name")
            if name_node is not None and name_node.text:
                detected_classes.add(name_node.text.strip())
    except Exception:
        continue

CLASSES = sorted(list(detected_classes))
print(f"Total de classes detectadas com sucesso: {len(CLASSES)}")
print("Exemplo de Classes:", CLASSES[:10])


def convert_xml_to_txt(xml_path, txt_output_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")

        if size is None:
            return False

        w = int(size.find("width").text)
        h = int(size.find("height").text)

        if w == 0 or h == 0:
            return False

        lines = []
        for obj in root.iter("object"):
            name_node = obj.find("name")
            if name_node is None or not name_node.text:
                continue

            cls_name = name_node.text.strip()
            if cls_name not in CLASSES:
                continue

            cls_id = CLASSES.index(cls_name)
            xmlbox = obj.find("bndbox")

            xmin = float(xmlbox.find("xmin").text)
            xmax = float(xmlbox.find("xmax").text)
            ymin = float(xmlbox.find("ymin").text)
            ymax = float(xmlbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2.0) / w
            y_center = ((ymin + ymax) / 2.0) / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            lines.append(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )

        if lines:
            with open(txt_output_path, "w") as out_file:
                out_file.writelines(lines)
            return True
        return False

    except Exception:
        return False


for split in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_OUTPUT_DIR, split, "labels"), exist_ok=True)

all_images = []
for xml_p in all_xml_files:
    base = os.path.splitext(xml_p)[0]
    for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
        img_p = base + ext
        if os.path.exists(img_p):
            all_images.append((img_p, xml_p))
            break

random.seed(42)
random.shuffle(all_images)

split_idx = int(len(all_images) * 0.8)
train_list = all_images[:split_idx]
val_list = all_images[split_idx:]


def build_yolo_dataset(data_list, split_name):
    print(f"Processando {split_name} dataset...")
    for img_path, xml_path in data_list:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        dest_img = os.path.join(
            YOLO_OUTPUT_DIR, split_name, "images", os.path.basename(img_path)
        )
        dest_txt = os.path.join(
            YOLO_OUTPUT_DIR, split_name, "labels", base_name + ".txt"
        )

        if convert_xml_to_txt(xml_path, dest_txt):
            shutil.copy(img_path, dest_img)


build_yolo_dataset(train_list, "train")
build_yolo_dataset(val_list, "val")

yaml_lines = [
    f"path: {YOLO_OUTPUT_DIR}",
    "train: train/images",
    "val: val/images",
    "names:",
]
for idx, cls_name in enumerate(CLASSES):
    yaml_lines.append(f"  {idx}: '{cls_name}'")

yaml_content = "\n".join(yaml_lines)
yaml_path = os.path.join(YOLO_OUTPUT_DIR, "data.yaml")

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("\nDataset pronto e estruturado para o YOLOv8!")

model = YOLO("yolov8n.pt")
gdrive_save_path = "/content/drive/MyDrive/yolov8_training_results"

model.train(
    data=yaml_path,
    epochs=50, # Ciclos de aprendizado
    imgsz=640,
    batch=16,
    device=0,
    project=gdrive_save_path, # Salvar checkpoints
    name="software_architecture_model",
)
