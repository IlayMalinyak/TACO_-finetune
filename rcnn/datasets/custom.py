"""
Custom dataset.

Mostly copy-paste from coco.py
"""
from pathlib import Path

from .coco import CocoDetection, make_coco_transforms

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} to custom dataset does not exist'
    training_json_file = 'custom_train.json'
    training_json_file_full = 'custom_train_full.json'
    validation_json_file = 'custom_val.json'
    test_json_file = 'custom_test.json'

    PATHS = {
        "train": (root , root / "annotations" / training_json_file),
        "val": (root , root / "annotations" / validation_json_file),
        "train_full": (root, root / "annotations" /training_json_file_full),
        "test": (root , root / "annotations" / test_json_file)
    }

    img_folder, ann_file = PATHS[image_set]
    print("img_folder", img_folder)
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
