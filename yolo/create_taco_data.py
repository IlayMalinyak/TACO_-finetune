import os
import shutil
import json
import random

images_new_dir = '/home/yanay.soker/mini_project/data'
images_source = '/datasets/TACO-master/data'

count_anns = {"train": 0, "val": 0, "test": 0}

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Read COCO format annotation file
def modify_and_copy(annotation_file, init_mode):
    """
    Read COCO format annotation file and change path to "train" or "val" directory
    """
    with open(annotation_file, 'r') as file:
        data = json.load(file)

    valid_data = []
    for i, d in enumerate(data['images']):
        if "dumped" not in d['file_name']:
            valid_data.append(i)
    print("num of valid data:", len(valid_data))
        
    create_dir_if_not_exists(f'{images_new_dir}/{init_mode}')
    create_dir_if_not_exists(f'{images_new_dir}/{init_mode}/imgs')
    create_dir_if_not_exists(f'{images_new_dir}/{init_mode}/anns')

    if init_mode=="train":
        create_dir_if_not_exists(f'{images_new_dir}/val')
        create_dir_if_not_exists(f'{images_new_dir}/val/imgs')
        create_dir_if_not_exists(f'{images_new_dir}/val/anns')

        n_to_select = int(len(valid_data)*0.2)
        val_sample = random.sample(valid_data, n_to_select)
    
    

    anns_of_img_by_id = dict()
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_of_img_by_id.keys():
            anns_of_img_by_id[img_id] = []
        anns_of_img_by_id[img_id].append(ann)

    for i, d in enumerate(data['images']):
        if i in valid_data:
            if init_mode=="train" and i in val_sample:
                mode = "val"
            else:
                mode = init_mode
                
            orig_path = d['file_name']
            id = d['id']
            new_img_path = os.path.join(f'{mode}', "imgs", f"{id}.jpg")
            d['file_name'] = new_img_path
            shutil.copy(f'{images_source}/{orig_path}', f'{images_new_dir}/{new_img_path}')
            # print(f"image copied from {images_source}/{orig_path} to {images_new_dir}/{new_img_path}")

            new_ann_path = os.path.join(f'{images_new_dir}', f'{mode}', "anns", f"{id}.txt")
            file = open(new_ann_path, "w", encoding="utf-8")
            im_height = d["height"]
            im_width = d["width"]
            for ann in anns_of_img_by_id[id]:
                x,y,width,height = [float(j) for j in ann['bbox']]
                cx = x+width/2
                cy = y+height/2
                category_id = ann['category_id']-1
                file.write(f"{category_id} {cx/im_width} {cy/im_height} {width/im_width} {height/im_height}\n")
                count_anns[mode]+=1
            file.close()
        
    # annotation_path = f'{root}/annotations/{mode}.json'
    # create_dir_if_not_exists(f'{root}/annotations')
    # with open(annotation_path, 'w') as file:
    #     json.dump(data, file, indent=2)


if __name__ == "__main__":
    modify_and_copy("annotations_train.json", "train")
    print("Done: train and val")
    modify_and_copy("annotations_test.json", "test")
    print("Done: test")
    print(count_anns)
