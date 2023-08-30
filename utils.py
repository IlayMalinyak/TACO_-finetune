import json
import matplotlib.pyplot as plt
import colorsys
from matplotlib.patches import Polygon, Rectangle
from PIL import Image, ExifTags
import os
import random
import numpy as np
from torchvision.ops import nms
from tqdm import tqdm
import torch
import pandas as pd
import seaborn as sns; sns.set()
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import fastrcnn_loss
import torch.nn.functional as F
import itertools
from functools import wraps
import inspect
from project.detr.util.box_ops import box_cxcywh_to_xyxy





cls_names = ('metals_and_plastic', 'other', 'non_recyclable', 'glass', 'paper', 'bio', 'unknown')

def filter_anns(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Step 2: Filter out elements with "data_captured" equal to 0
    data['images'] = [image for image in data['images'] if image.get('coco_url', 1) != '']

    # Step 3: Save the modified JSON back to the same file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def fill_anns(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Step 2: Fill dummy segmentation
    for d in data['annotations']:
        if 'segmentation' not in d.keys():
            d['segmentation'] = []

    # Step 3: Save the modified JSON back to the same file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def anns_hist(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    anns = data['annotations']
    images_ids = set([ann['image_id'] for ann in anns])
    images = [image for image in data['images']]
    nr_imgs = len(images)
    nr_imgs_no_id = 0
    for image in images:
        if image['id'] not in images_ids:
            nr_imgs_no_id += 1
    print(f'Number of images: {nr_imgs}')
    print(f'Number of images with no annotations: {nr_imgs_no_id}')
    categories = data['categories']
    nr_cats = len(categories) 
    cat_histogram = np.zeros(nr_cats,dtype=int)
    for ann in anns:
        cat_histogram[ann['category_id']-1] += 1
    
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(15,5))
    
    # Convert to DataFrame
    df = pd.DataFrame({'Categories': list(cls_names), 'Number of annotations': cat_histogram})
    df['inv_density'] = df['Number of annotations'].sum() / (df['Number of annotations']+1) 
    df['weights'] = df['inv_density'] / df['inv_density'].sum()
    df = df.sort_values('Number of annotations', axis=0, ascending=False)
    print("annotations DataFrame:\n", df)
    
    # Plot the histogram
    # sns.set_color_codes("pastel")
    # sns.set(style="whitegrid")
    plot_1 = sns.barplot(y="Number of annotations", x="Categories", data=df,
                label="Total", color="b", orient='v', order=df['Categories'])
    
    fig = plot_1.get_figure()
    plt.show()
    # fig.savefig("output.png")


def count_imgs(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    images = [image for image in data['images']]
    return len(images)


def show_sample(img, boxes, labels, color=None, ax=None):
    if ax is None:
        fig,ax = plt.subplots(1)
    plt.axis('off')
    ax.imshow(img)
    for b in range(len(boxes)):
        color = colorsys.hsv_to_rgb(np.random.random(),1,1) if color is None else color
        (x1, y1, x2, y2) = boxes[b]
        w,h = x2-x1, y2-y1
        rect = Rectangle((x1,y1),w,h,linewidth=2,edgecolor=color,
                         facecolor='none', alpha=0.7, linestyle = '--')
        label = cls_names[labels[b].item()]
        plt.gca().text(x1+10, y1+10, label, color=color, fontsize=10)
        plt.add_patch(rect)
    return ax


def split_coco_annotation_file(full_annotation_file, train_file, test_file, split_ratio):
    # Load the full COCO annotation file
    with open(full_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Shuffle the images to split randomly
    random.seed(123)  # For reproducibility, you can change the seed
    random.shuffle(coco_data['images'])

    # Calculate the number of images for the train and test splits
    num_images = len(coco_data['images'])
    num_train_images = int(num_images * split_ratio)
    num_test_images = num_images - num_train_images

    # Split the images into train and test sets
    train_images = coco_data['images'][:num_train_images]
    test_images = coco_data['images'][num_train_images:]

    # Filter annotations based on the split images
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in set(img['id'] for img in train_images)]
    test_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in set(img['id'] for img in test_images)]

    # Create train and test annotation dictionaries
    train_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories']
    }

    test_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': test_images,
        'annotations': test_annotations,
        'categories': coco_data['categories']
    }

    # Save the train and test annotation files
    with open(train_file, 'w') as f_train:
        json.dump(train_data, f_train)

    with open(test_file, 'w') as f_test:
        json.dump(test_data, f_test)


if __name__ == "__main__":
    anns_hist("annotations_train_full.json")