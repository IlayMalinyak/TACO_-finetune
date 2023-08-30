import os
import shutil
import json

root = '/home/ilay.kamai/mini_project/data'
old_root = '/datasets/TACO-master/data'

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Read COCO format annotation file
def modify_and_copy(annotation_file, mode):
    """
    Read COCO format annotation file and change path to "train2017" or "val2017" directory
    """
    create_dir_if_not_exists(f'{root}/{mode}2017')
    with open(annotation_file, 'r') as file:
        data = json.load(file)
    for i, d in enumerate(data['images']):
        orig_path = d['file_name']
        new_path = os.path.join(f'{mode}2017', f"{i}.jpg")
        d['file_name'] = new_path
        shutil.copy(f'{old_root}/{orig_path}', f'{root}/{new_path}')
        print(f"image copied from {old_root}/{orig_path} to {root}/{new_path}")
    annotation_path = f'{root}/annotations/{mode}.json'
    create_dir_if_not_exists(f'{root}/annotations')
    with open(annotation_path, 'w') as file:
        json.dump(data, file, indent=2)


if __name__ == "__main__":
    create_dir_if_not_exists(root)
    # modify_and_copy("annotations_train.json", "train")
    # modify_and_copy("annotations_val.json", "val")
    # modify_and_copy("annotations_test.json", "test")
    modify_and_copy("annotations_train_full.json", "train_full")