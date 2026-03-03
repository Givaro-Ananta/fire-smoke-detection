"""
Fire & Smoke Dataset Preparation
- Preprocessing: resize, validate, normalize
- Convert Pascal VOC XML to YOLO format
- Split into train/val (80/20)
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


# ========================
# CONFIGURATION
# ========================
BASE_DIR = os.path.dirname(__file__)
XML_DIR = os.path.join(BASE_DIR, "fire_smoke")

# Look for images in these directories in addition to XML_DIR
# Note: user pointed out they have fire_smoke-9 and fire_smoke-10 folders
IMAGE_DIRS = [
    os.path.join(BASE_DIR, "fire_smoke"),
    os.path.join(BASE_DIR, "fire_smoke-9"),
    os.path.join(BASE_DIR, "fire_smoke-10")
]

OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")
IMG_SIZE = 640
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# ========================
# PREPROCESSING FUNCTIONS
# ========================

def validate_image(img_path):
    """Check if image is valid and not corrupt."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        if img.shape[0] == 0 or img.shape[1] == 0:
            return False
        return True
    except Exception:
        return False


def preprocess_image(img_path, output_path, target_size=IMG_SIZE):
    """
    Preprocess image:
    - Resize to target_size x target_size
    - Normalize brightness/contrast (CLAHE)
    - Save to output_path
    Returns (new_width, new_height) or None if failed.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # --- Brightness/Contrast normalization using CLAHE ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- Resize to target size (letterbox to preserve aspect ratio) ---
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas and center the image
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img_resized

    cv2.imwrite(output_path, canvas)

    return w, h, new_w, new_h, left, top


# ========================
# VOC TO YOLO CONVERSION
# ========================

def parse_voc_xml(xml_path):
    """Parse Pascal VOC XML and return list of (class_name, xmin, ymin, xmax, ymax)."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        # Catch any parsing or file error
        return None, None, None

    # Get image size
    size_elem = root.find("size")
    if size_elem is None:
        return None, None, None

    width = int(size_elem.find("width").text)
    height = int(size_elem.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        objects.append((name, xmin, ymin, xmax, ymax))

    return objects, width, height


def voc_to_yolo(objects, img_w, img_h, class_map, 
                target_size=IMG_SIZE, orig_w=None, orig_h=None,
                new_w=None, new_h=None, pad_left=0, pad_top=0):
    """
    Convert VOC bounding boxes to YOLO format with letterbox adjustment.
    Returns list of (class_id, cx, cy, w, h) normalized to [0, 1].
    """
    yolo_labels = []

    # Calculate scale from original to resized
    if orig_w and orig_h and new_w and new_h:
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
    else:
        scale_x = target_size / img_w
        scale_y = target_size / img_h
        pad_left = 0
        pad_top = 0

    for name, xmin, ymin, xmax, ymax in objects:
        if name not in class_map:
            continue

        class_id = class_map[name]

        # Adjust coordinates for letterbox
        new_xmin = xmin * scale_x + pad_left
        new_ymin = ymin * scale_y + pad_top
        new_xmax = xmax * scale_x + pad_left
        new_ymax = ymax * scale_y + pad_top

        # Convert to YOLO format (center x, center y, width, height) normalized
        cx = ((new_xmin + new_xmax) / 2.0) / target_size
        cy = ((new_ymin + new_ymax) / 2.0) / target_size
        w = (new_xmax - new_xmin) / target_size
        h = (new_ymax - new_ymin) / target_size

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        if w > 0.001 and h > 0.001:
            yolo_labels.append((class_id, cx, cy, w, h))

    return yolo_labels


# ========================
# MAIN PIPELINE
# ========================

def discover_classes(xml_dir):
    """Scan all XML files to discover unique class names."""
    classes = set()
    xml_files = list(Path(xml_dir).glob("*.xml"))
    print(f"Scanning {len(xml_files)} XML files for class names...")

    for xml_file in tqdm(xml_files, desc="Discovering classes"):
        objects, _, _ = parse_voc_xml(str(xml_file))
        if objects:
            for name, *_ in objects:
                classes.add(name)

    class_list = sorted(list(classes))
    print(f"Found {len(class_list)} classes: {class_list}")
    return class_list


def find_image_for_xml(xml_path, image_dirs):
    """Find the corresponding image file for an XML annotation across multiple directories."""
    img_filename = None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename_elem = root.find("filename")
        if filename_elem is not None:
            img_filename = filename_elem.text
    except Exception:
        pass

    # Method 1: Try using the filename specified in the XML across all image dirs
    if img_filename:
        for img_dir in image_dirs:
            if not os.path.exists(img_dir):
                continue
            img_path = os.path.join(img_dir, img_filename)
            if os.path.exists(img_path):
                return img_path

    # Method 2: Fallback - try replacing .xml with common image extensions across all image dirs
    base = os.path.splitext(os.path.basename(xml_path))[0]
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        for img_dir in image_dirs:
            if not os.path.exists(img_dir):
                continue
            img_path = os.path.join(img_dir, base + ext)
            if os.path.exists(img_path):
                return img_path

    return None


def prepare_dataset():
    """Main function to prepare the dataset."""
    print("=" * 60)
    print("Fire & Smoke Dataset Preparation")
    print("=" * 60)

    # Step 1: Discover classes
    class_list = discover_classes(XML_DIR)
    class_map = {name: idx for idx, name in enumerate(class_list)}

    # Step 2: Create output directories
    splits = ["train", "val"]
    for split in splits:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

    # Step 3: Collect all valid XML-image pairs
    xml_files = list(Path(XML_DIR).glob("*.xml"))
    print(f"\nProcessing {len(xml_files)} annotation files...")

    valid_pairs = []
    skipped_no_image = 0
    skipped_corrupt = 0
    skipped_no_objects = 0

    for xml_file in tqdm(xml_files, desc="Validating pairs"):
        xml_path = str(xml_file)

        # Find corresponding image
        img_path = find_image_for_xml(xml_path, IMAGE_DIRS)
        if img_path is None:
            skipped_no_image += 1
            continue

        # Validate image
        if not validate_image(img_path):
            skipped_corrupt += 1
            continue

        # Parse XML and check for objects
        objects, img_w, img_h = parse_voc_xml(xml_path)
        if objects is None or len(objects) == 0:
            skipped_no_objects += 1
            continue

        valid_pairs.append((xml_path, img_path, objects, img_w, img_h))

    print(f"\nValidation Summary:")
    print(f"   Valid pairs: {len(valid_pairs)}")
    print(f"   Skipped (no image): {skipped_no_image}")
    print(f"   Skipped (corrupt image): {skipped_corrupt}")
    print(f"   Skipped (no objects): {skipped_no_objects}")

    # Step 4: Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(valid_pairs)

    split_idx = int(len(valid_pairs) * TRAIN_RATIO)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"\nSplit:")
    print(f"   Train: {len(train_pairs)} images")
    print(f"   Val:   {len(val_pairs)} images")

    # Step 5: Process and save
    class_counts = {name: {"train": 0, "val": 0} for name in class_list}

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        print(f"\nProcessing {split_name} set...")

        for xml_path, img_path, objects, img_w, img_h in tqdm(pairs, desc=f"  {split_name}"):
            base_name = Path(img_path).stem
            out_img_path = os.path.join(OUTPUT_DIR, split_name, "images", f"{base_name}.jpg")
            out_lbl_path = os.path.join(OUTPUT_DIR, split_name, "labels", f"{base_name}.txt")

            # Preprocess image (resize + normalize)
            result = preprocess_image(img_path, out_img_path, IMG_SIZE)
            if result is None:
                continue

            orig_w, orig_h, new_w, new_h, pad_left, pad_top = result

            # Convert labels
            yolo_labels = voc_to_yolo(
                objects, img_w, img_h, class_map,
                IMG_SIZE, orig_w, orig_h, new_w, new_h, pad_left, pad_top
            )

            # Write YOLO label file
            with open(out_lbl_path, "w") as f:
                for class_id, cx, cy, w, h in yolo_labels:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            # Count classes
            for name, *_ in objects:
                if name in class_counts:
                    class_counts[name][split_name] += 1

    # Step 6: Generate data.yaml
    yaml_path = os.path.join(os.path.dirname(__file__), "data.yaml")
    dataset_abs = os.path.abspath(OUTPUT_DIR)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_abs}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"\n")
        f.write(f"nc: {len(class_list)}\n")
        f.write(f"names: {class_list}\n")

    # Step 7: Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Total':<10}")
    print("-" * 50)
    for name in class_list:
        train_c = class_counts[name]["train"]
        val_c = class_counts[name]["val"]
        print(f"{name:<20} {train_c:<10} {val_c:<10} {train_c + val_c:<10}")
    print("-" * 50)
    print(f"{'TOTAL':<20} {len(train_pairs):<10} {len(val_pairs):<10} {len(valid_pairs):<10}")
    print(f"\nDataset saved to: {dataset_abs}")
    print(f"Config saved to: {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_dataset()
