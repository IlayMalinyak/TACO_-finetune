import pycocotools.coco as coco
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image, ExifTags 
import subprocess
import os
import json

import sys
# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the parent directory of the script to the Python path
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)


dataDir='/home/ilay.kamai/mini_project/data'
dataType=''
annFile='{}/annotations/custom_train.json'.format(dataDir)

num_classes = 8

finetuned_classes = [
      'N/A', 'metals_and_plastic', 'other', 'non_recyclable', 'glass', 'paper', 'bio', 'unknown', 'background'
  ]

def finetune():
    # alpha_t_list = [0,0.006, 0.22, 0.014, 0.03, 0.063, 0.6, 0.01, 0.005] # train inverse frequencies
    # alpha_t_list = [0,0.01, 0.36, 0.023, 0.05, 0.105, 1, 0.016, 0.008] # train inverse frequencies normalized
    # alpha_t_list = [0,0.7, 1, 0.7, 1, 1, 1, 1, 0.0005]
    alpha_t_list = [0,0.7,1,0.8,1,1,1,1,0.0005]


    # Convert the list to a JSON-formatted string
    alpha_t_str = " ".join(map(str, alpha_t_list))
    # Define the command as a list of arguments
    command = [
        "python", "submission/project/detr/main.py",   
        "--dataset_file", "custom",
        "--coco_path", "/home/ilay.kamai/mini_project/data",
        "--output_dir", "detr/outputs_test3",
        "--resume", "detr/detr-r50_no-class-head.pth",
        # "--freeze_backbone", "True",
        # "--resume", "detr/outputs3/checkpoint.pth",
        "--num_classes", str(num_classes),  # Replace num_classes with the actual value you have
        "--batch_size", "2",
        "--eos_coef", "0.0005",
        "--epochs", "50",
        "--weight_decay", "8e-4",
        "--early_stopping", "10",
        # "--bbox_loss_coef", "1",
        # "--giou_loss_coef", "2",
        # "--ce_loss_coef", "2",
        "--alpha_t", *alpha_t_str.split(),
        # "--gamma", "1.5",
        # "--focal_loss", "True",

    ]
    try:
        print("start training with command:", " ".join(command))
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Output:", result.stdout)
        print("Error:", result.stderr)
        print("finished with return code:", result.returncode)
    except subprocess.CalledProcessError as e:
        print("Command failed with exit code:", e.returncode)
        print("Error:", e.stderr)
    # Execute the command


if __name__ == '__main__':
    finetune()
    
