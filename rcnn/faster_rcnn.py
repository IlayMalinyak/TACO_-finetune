import torch, torchvision
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as t
import albumentations as A
import json
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP
from matplotlib import pyplot as plt


import sys
# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the parent directory of the script to the Python path
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# from detr.detr_predict import calculate_mAP
from datasets.coco import build as build_dataset
from util import misc as d_utils
from util.plot_utils import plot_fit, plot_logs
from pathlib import Path

import os
from rcnn_train import Trainer
# from ..utils import Cfg, plot_fit



def finetune():
    print(os.system('nvidia-smi'))

    torch.manual_seed(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # img_file = '/datasets/TACO-master/data/'
    # ann_file = '/home/ilay.kamai/mini_project/annotations_{}.json'
    n_classes = 8
    exp_num=0
    cls_names = [
        'N/A', 'metals_and_plastic', 'other', 'non_recyclable', 'glass', 'paper', 'bio', 'unknown', 'background'
    ]
    args = d_utils.Cfg(device=device, batch_size=2, lr=1e-4, weight_decay=1e-4, coco_path='/home/ilay.kamai/mini_project/data',
    output_dir='/home/ilay.kamai/mini_project/rcnn_output_test')


    dataset_train = build_dataset(image_set='train_full', args=args)
    dataset_val = build_dataset(image_set='test', args=args)
    dataset_test = build_dataset(image_set='test', args=args)


    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    train_dl = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=d_utils.collate_fn, num_workers=args.num_workers)
    val_dl = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=d_utils.collate_fn, num_workers=args.num_workers)
    test_dl = DataLoader(dataset_test, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=d_utils.collate_fn, num_workers=args.num_workers)


    torchvision.models.detection.roi_heads.fastrcnn_loss = d_utils.custom_fastrcnn_loss

    optim_param = {'lr':args.lr, 'weight_decay':args.weight_decay}
    # Model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1

    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )

    # anchor_generator = AnchorGenerator(sizes=((16, 32, 64),),
    #                                aspect_ratios=((0.5, 1.0, 2.0),))
    # # print(model)

    model.rpn.anchor_generator = anchor_generator

    # # 256 because that's the number of features that FPN returns
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    # trainables = [name for name, p in model.named_parameters() if p.requires_grad]
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters}')

    optimizer = torch.optim.AdamW(model.parameters(), **optim_param)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True, factor=0.1)
                
    trainer = Trainer(model=model, optimizer=optimizer, criterion=None,
                        scheduler=scheduler, train_dl=train_dl, val_dl=val_dl,
                            device=device, optim_params=optim_param, net_params=None, nms_iou_thresh=0.6, exp_num=exp_num, log_path=args.output_dir,
                            exp_name="faster_rcnn", metric=MAP, validate=False)

    model = model.to(device)
    res = trainer.fit(num_epochs=15, device=device, early_stopping=15, best='loss')

    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'checkpoint.pth')))
    preds, targets = trainer.predict(model, test_dl, device=device, cls_names=cls_names, plot_every=30, save_path=args.output_dir)


    # out_dir2 = '/home/ilay.kamai/mini_project/rcnn_output2'
    log_directory = [Path(args.output_dir)]
    fields_of_interest = (
            'mAP',
            'loss',
            )
    # names = ['focal', 'smoothed ce']
    plot_logs(log_directory,
            fields_of_interest)
    plt.savefig(f'{args.output_dir}/mAP_loss_test.png')
    plt.clf()

    metric = MAP(iou_thresholds=[0.25,0.5,0.75], rec_thresholds=[0.25,0.5,0.75], class_metrics=True)
    acc = metric(preds, targets)
    acc_dict = {k:v.tolist() for k,v in acc.items() }
    acc_path = f'{args.output_dir}/test_acc.json'
    with open(acc_path, 'w') as file:
            json.dump(acc_dict, file, indent=2)

    all_labels = []
    all_gt_labels = []
    # cls_names.append('background')
    for i in range(len(targets)):
        if len(targets[i]['labels']):
            all_gt_labels.extend(targets[i]['labels'].cpu().tolist())
        else:
            all_gt_labels.append(8)
        all_labels.extend(preds[i]['labels'].cpu().tolist())
    bin_edges = np.histogram_bin_edges(np.concatenate([all_labels, all_gt_labels]), bins='auto')
    plt.hist(all_gt_labels, label='True', bins=bin_edges)
    plt.hist(all_labels, alpha=0.5, label='Predicted', bins=bin_edges)
    plt.xticks(range(len(cls_names)), cls_names, size='small', rotation='vertical', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("classes distribution - test set", fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(f'{args.output_dir}/cls_dist.jpeg')
    plt.clf()


    # plot_fit(res, train_test_overlay=True)
    # plt.savefig(f'{args.output_dir}/fit.jpeg')
    # plt.clf()

    print("finished predictions")

if __name__ == '__main__':
    finetune()
