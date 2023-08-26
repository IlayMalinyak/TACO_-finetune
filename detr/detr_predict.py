import sys
import os
# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the parent directory of the script to the Python path
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from detr.datasets.coco import build as build_dataset
import torch
from torch.utils.data import DataLoader
import util.misc as d_utils
from util.plot_utils import plot_logs, plot_images_with_bboxes, plot_attention_map
from util.box_ops import box_cxcywh_to_xyxy, rescale_bboxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP
import json
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
import detr_finetune as detr_finetune


def calculate_ap(precision, recall):
    """
    Calculate Average Precision (AP) for a single class.

    Parameters:
        precision (List[float]): List of precision values for the class.
        recall (List[float]): List of recall values for the class.

    Returns:
        float: Average Precision (AP) for the class.
    """
    # Ensure the precision and recall lists are not empty
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    # Prepend 0 to precision and recall lists to make them start at 0.0 recall
    precision = [0.0] + precision + [0.0]
    recall = [0.0] + recall + [1.0]

    # Compute the precision at each recall value
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Compute the area under the precision-recall curve (AP)
    ap = 0.0
    for i in range(len(recall) - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap



def calculate_iou(box1, box2, convert1=False, convert2=True):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        box1 (List[float]): List containing the coordinates [x_min, y_min, x_max, y_max] of the first box.
        box2 (List[float]): List containing the coordinates [x_min, y_min, x_max, y_max] of the second box.

    Returns:
        float: Intersection over Union (IoU) between the two boxes.
    """
    if convert1:
        box1 = box_cxcywh_to_xyxy(torch.tensor(box1)).tolist()
    if convert2:
        box2 = box_cxcywh_to_xyxy(torch.tensor(box2)).tolist()
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)

    inter_area = max(0, inter_x_max - inter_x_min + 1) * max(0, inter_y_max - inter_y_min + 1)
    box1_area = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1)
    box2_area = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def calculate_mAP(pred_targets, true_targets, iou_threshold=0.5, num_classes=8, convert1=False, convert2=True):
    """
    Calculate mean Average Precision (mAP) for object detection.

    Parameters:
        pred_targets (List[Dict[str, Tensor]]): List of predicted targets for each image.
                                                Each dictionary contains 'boxes' and 'labels' keys.
        true_targets (List[Dict[str, Tensor]]): List of ground truth targets for each image.
                                                Each dictionary contains 'boxes' and 'labels' keys.
        iou_threshold (float): IoU threshold to consider a detection as correct (default: 0.5).
        num_classes (int): Number of classes in the dataset (default: 20).

    Returns:
        float: Mean Average Precision (mAP) across all classes.
    """
    average_precisions = []

    for class_idx in range(1, num_classes):
        true_positives = []
        false_positives = []
        num_gt_targets = 0

        for pred_targets_per_image, true_targets_per_image in zip(pred_targets, true_targets):
            true_targets_class = []
            scores = []
            pred_targets_class = []
            for i in range(len(true_targets_per_image['boxes'])):
                if true_targets_per_image['labels'][i] == class_idx:
                    true_targets_class.append(true_targets_per_image['boxes'][i].tolist())
            for i in range(len(pred_targets_per_image['boxes'])):
                if pred_targets_per_image['labels'][i] == class_idx:
                    pred_targets_class.append(pred_targets_per_image['boxes'][i].tolist())
                    scores.append(pred_targets_per_image['scores'][i])

            # pred_targets_class = [target for target in pred_targets_per_image['boxes'] if target["labels"] == class_idx]
            # true_targets_class = [target for target in true_targets_per_image['boxes'] if target["labels"] == class_idx]
            num_gt_targets += len(true_targets_class)

            # Sort predicted targets by confidence score in descending order
            # scores = pred_targets_per_image.get('scores', None)
            if len(scores):
                sorted_indices = torch.argsort(torch.tensor(scores), descending=True)
                pred_targets_class = [pred_targets_class[i] for i in sorted_indices]

                detected_targets = [0] * len(true_targets_class)  

                for pred_idx, pred_target in enumerate(pred_targets_class):
                    if len(true_targets_class):
                        overlaps = [calculate_iou(pred_target, true_target, convert1, convert2) for true_target in true_targets_class]
                        max_iou_idx = overlaps.index(max(overlaps))
                        if overlaps[max_iou_idx] >= iou_threshold and not detected_targets[max_iou_idx]:
                            true_positives.append(1)
                            detected_targets[max_iou_idx] = 1
                        else:
                            # print("iou: ", overlaps[max_iou_idx], "thresh: ",  iou_threshold)
                            false_positives.append(1)
        # print(f"Class {class_idx}: {len(true_positives)} true positives, {len(false_positives)} false positives, {num_gt_targets} ground truth targets")
        # Compute precision and recall
        tp_cumsum = torch.cumsum(torch.tensor(true_positives), dim=0)
        fp_cumsum = torch.cumsum(torch.tensor(false_positives), dim=0)

        # Pad tp_cumsum and fp_cumsum with zeros to have the same length
        max_length = max(tp_cumsum.shape[0], fp_cumsum.shape[0])
        pad_length_tp = max_length - tp_cumsum.shape[0]
        pad_length_fp = max_length - fp_cumsum.shape[0]
        if pad_length_tp > 0:
            tp_cumsum = torch.cat((tp_cumsum, torch.zeros(pad_length_tp)))
        if pad_length_fp > 0:
            fp_cumsum = torch.cat((fp_cumsum, torch.zeros(pad_length_fp)))

        recall = tp_cumsum.float() / num_gt_targets
        precision = tp_cumsum.float() / (tp_cumsum + fp_cumsum)

        # Calculate AP for the class and store it in the list
        ap = calculate_ap(precision.tolist(), recall.tolist())
        # if num_gt_targets:
        average_precisions.append(ap)
    # print("average precisions ", average_precisions)
    # Calculate mAP by taking the mean of all class APs
    mAP = sum(average_precisions) / len(average_precisions)
    return mAP

def result_dict_from_logits(imgs, targets, proposal_boxes, proposal_logits, batch_indices, box_indices, dec_attn=None,
                             enc_attn=None):       
        chosen_boxes_list = []
    
        for batch_idx in range(len(proposal_boxes)):
            # Find matching indices for this batch_idx
            matching_indices = np.where(batch_indices == batch_idx)[0]
        
            # If there are matching indices, gather the chosen boxes, logits, and probabilities
            if len(matching_indices) > 0:
                img_h, img_w = imgs[batch_idx].shape[1:]
                # print(img_w, img_h)
                chosen_boxes = proposal_boxes[batch_idx][box_indices[matching_indices]]
                chosen_boxes = torch.stack([rescale_bboxes(box, (img_w, img_h)) for box in chosen_boxes])
                targets[batch_idx]['boxes'] = torch.stack([rescale_bboxes(box, (img_w, img_h)) for box in targets[batch_idx]['boxes']])
                d_attn = dec_attn[batch_idx][box_indices[matching_indices]] if dec_attn is not None else torch.empty((0), dtype=torch.float32)
                e_attn = enc_attn[batch_idx][box_indices[matching_indices]] if enc_attn is not None else torch.empty((0), dtype=torch.float32)
                # chosen_boxes = convert_box_01_to_image_size(box_cxcywh_to_xyxy(chosen_boxes), img_w, img_h)
                logits = proposal_logits[batch_idx][box_indices[matching_indices]]
                prob = F.softmax(logits, -1)
                scores, labels = prob.max(-1)
            else:
                # If no matching indices, create empty tensors
                chosen_boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty(0, dtype=torch.int64)
                scores = torch.empty(0, dtype=torch.float32)
                d_attn = torch.empty((0), dtype=torch.float32)
                e_attn = torch.empty((0), dtype=torch.float32)
            
            # Create a dictionary for this batch_idx
            batch_dict = {
                'boxes': chosen_boxes,
                'labels': labels,
                'scores': scores,
                'dec_attn': d_attn,
                'enc_attn': e_attn
            }
        
            # Append the dictionary to the chosen_boxes_list
            chosen_boxes_list.append(batch_dict)
        
        return chosen_boxes_list, targets

@torch.no_grad()
def predict(model, test_dl, criterion, device, plot_every=None, save_path=None ):
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]    
    model.eval()
    tot_targets = []
    tot_gt_targets = []
    with tqdm(test_dl, desc=f"Predict Loss: ", unit="batch") as pbar:
        for it, (samples, targets) in enumerate(pbar):
            plot = (plot_every is not None) and (it % plot_every == 0)
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # use lists to store the outputs via up-values
            outputs = model(samples)
            for hook in hooks:
                hook.remove()
            # print("pred logits: ", outputs["pred_logits"].shape)
            # conv_features = conv_features[0]
            # enc_attn_weights = enc_attn_weights[0]
            # dec_attn_weights = dec_attn_weights[0]
            results = [{}]*len(targets)
            indices = criterion.matcher(outputs, targets)
            src_indices = criterion._get_src_permutation_idx(indices)
            results, targets = result_dict_from_logits(samples.tensors, targets, outputs['pred_boxes'],
                                                        outputs["pred_logits"], src_indices[0], src_indices[1],
                                                          dec_attn=dec_attn_weights[0],enc_attn=enc_attn_weights[0])
            tot_gt_targets.extend(targets)
            tot_targets.extend(results)
            if plot:
                targets_np = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in targets]
                pred_np = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in results]
                images_np = samples.tensors.detach().cpu().numpy()
                plot_images_with_bboxes(images_np, pred_np, detr_finetune.finetuned_classes, gt_target=targets_np, save_path=save_path,
                                            name=f"preds_{it}")
                plot_attention_map(images_np, pred_np, conv_features[0], detr_finetune.finetuned_classes, save_path=save_path, name=f"attn_{it}")
    return tot_targets, tot_gt_targets 

def predict_all(args, model, criterion, postprocessors, test=False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)



    # args = utils.DetrCfg(device=device, batch_size=1, lr=2e-3, output_dir=LOG_PATH)

    log_directory = [Path(args.output_dir)]

    for (att1, att2) in zip(['mAP', 'class_error'], ['loss', 'cardinality_error_unscaled']):
        plot_logs(log_directory,
                (att1, att2))
        plt.savefig(f'{args.output_dir}/{att1}_{att2}.png')

    fields_of_interest = (
        'loss_ce',
        'loss_bbox',
        'loss_giou',
        )
    plot_logs(log_directory,
            fields_of_interest)
    plt.savefig(f'{args.output_dir}/losses.png')

    if test:
        print("evaluating on test set...")
        # model, criterion, postprocessors  = build_model(args)
        checkpoint = torch.load(f"{args.output_dir}/checkpoint.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)


        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


        dataset_test = build_dataset(image_set='test', args=args)

        # base_ds = dataset_val.coco

        sampler_val = torch.utils.data.SequentialSampler(dataset_test)

        test_dl = DataLoader(dataset_test, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=d_utils.collate_fn, num_workers=args.num_workers)

        optim_params = {'lr': args.lr, 'weight_decay':args.weight_decay}
    

        model = model.to(device)

        print('loading best model')
        model.load_state_dict(torch.load(args.output_dir + '/checkpoint.pth')['model'])
        print(os.system('nvidia-smi'))
        print("predicting...")
        preds, targets = predict(model, test_dl, criterion, device, plot_every=30, save_path=args.output_dir)
        targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
        preds = [{k: v.cpu() for k, v in t.items()} for t in preds]

        map_thresh = [0.25,0.5,0.75]
        metric = MAP(iou_thresholds=map_thresh, rec_thresholds=map_thresh, class_metrics=True)
        fig_, ax_ = metric.plot()
        plt.savefig(f'{args.output_dir}/test_mAP.png')
        acc = metric(preds, targets)
        acc_dict = {k:v.tolist() for k,v in acc.items() }
        acc_path = f'{args.output_dir}/test_acc.json'
        with open(acc_path, 'w') as file:
                json.dump(acc_dict, file, indent=2)

        # acc_ = {k:v.item() for k,v in acc.items() if k != 'classes'}
        # acc_path2 = f'{args.output_dir}/test_acc2.json'
        # acc2 = [calculate_mAP(preds, targets, iou_threshold=thresh, convert2=False) for thresh in map_thresh]
        # acc2_dict = {f'mAP_{thresh}': acc2[i] for i, thresh in enumerate(map_thresh)}
        # with open(acc_path2, 'a') as file:
        #         json.dump(acc2_dict, file, indent=2)
        # print("accuracy: ", acc_)
        # print("scratch: ", acc2, np.mean(acc2))

        plt.close('all') 
        all_labels = []
        all_gt_labels = []
        cls_names = detr_finetune.finetuned_classes
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
        plt.tight_layout()
        plt.savefig(f'{args.output_dir}/cls_dist.jpeg')

    print("finished predictions")