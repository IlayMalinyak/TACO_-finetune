from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import time
import os
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchmetrics.detection.mean_ap import MeanAveragePrecision as MAP
from torchvision.ops import nms
import math
import detr.util.misc as d_utils
from util.misc import initializer
from util.plot_utils import plot_images_with_bboxes
from torchvision.ops import nms
import torch.nn.functional as F
from util.box_ops import box_cxcywh_to_xyxy, rescale_bboxes
import sys
import json
from pathlib import Path




def rcnn_forward(model, images, targets):
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    # model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    model.roi_heads.training=True


    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    # print("class_logits", class_logits.shape, "image_shapes", image_shapes, "boxes:", box_regression.shape)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "pred_logits": class_logits[i],
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections


class Trainer(object):
    """
    A class that encapsulates the training loop for a PyTorch model.
    """
    @initializer
    def __init__(self, model, optimizer, criterion, metric, scheduler, train_dl, val_dl, device, optim_params, net_params, exp_num, log_path,
                 exp_name,plot_every=None, max_iter=math.inf, nms_iou_thresh=0.7, max_norm=0, postprocessors=None,base_ds=None, output_dir=None,
                   coco_evaluator=None, validate=True):
        if log_path is not None:
            if  not os.path.exists(f'{self.log_path}'):
                os.makedirs(f'{self.log_path}')
            # with open(f'{self.log_path}/exp{exp_num}/net_params.yml', 'w') as outfile:
            #     yaml.dump(self.net_params, outfile, default_flow_style=False)
            # with open(f'{self.log_path}/exp{exp_num}/optim_params.yml', 'w') as outfile:
            #         yaml.dump(self.optim_params, outfile, default_flow_style=False)


    def fit(self, num_epochs, device,  early_stopping=None, best='loss'):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = np.inf
        best_acc = 0
        train_loss, val_loss,  = [], []
        train_acc, val_acc = [], []
        self.optim_params['lr_history'] = []
        epochs_without_improvement = 0
        v_acc=0

        print(f"Starting training for {num_epochs} epochs with parameters: {self.optim_params}, {self.net_params}")
        for epoch in range(num_epochs):
            start_time = time.time()
            plot = (self.plot_every is not None) and (epoch % self.plot_every == 0)
            t_stats = self.train_epoch(device, epoch=epoch, plot=plot, max_iter=self.max_iter)
            t_loss_mean = t_stats['loss']
            # self.logger.add_scalar('train_acc', t_acc, epoch)
            # train_acc.append(t_acc)
            # train_loss.extend(t_loss)
            if self.validate:
                v_stats = self.eval_epoch(device, epoch=epoch, plot=plot, max_iter=self.max_iter)
                v_loss_mean = v_stats['loss']
                v_acc = v_stats['mAP']
            else:
                v_stats = {}
                v_loss_mean = np.nan
                v_acc = np.nan

            # self.logger.add_scalar('validation_acc', v_acc, epoch)
            # val_acc.append(v_acc)
            # val_loss.extend(v_loss)
            log_stats = {**{f'train_{k}': v for k, v in t_stats.items()},
                     **{f'test_{k}': v for k, v in v_stats.items()},
                     'epoch': epoch,
                     }

            if self.log_path and d_utils.is_main_process():
                with (Path(f"{self.log_path}/log.txt")).open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            if self.validate:
                if self.scheduler is not None:
                    self.scheduler.step(v_loss_mean)
                criterion = min_loss if best == 'loss' else best_acc
                mult = 1 if best == 'loss' else -1
                objective = v_loss_mean if best == 'loss' else v_acc
                if mult*objective < mult*criterion:
                    print(f"saving model at {self.log_path}/checkpoint.pth")
                    if best == 'loss':
                        min_loss = v_loss_mean
                    else:
                        best_acc = v_acc
                    torch.save(self.model.state_dict(), f'{self.log_path}/checkpoint.pth')
                    self.best_state_dict = self.model.state_dict()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement == early_stopping:
                        print('early stopping!', flush=True)
                        break
            else:
                torch.save(self.model.state_dict(), f'{self.log_path}/checkpoint.pth')
                self.best_state_dict = self.model.state_dict()
                
            # self.logger.add_scalar('time', time.time() - start_time, epoch)
            print(f'****Epoch {epoch}: Train Loss: {t_loss_mean:.6f}, Val Loss: {v_loss_mean:.6f}, Train Acc: {0:.6f}, Val Acc: {v_acc:.6f}, Time: {time.time() - start_time:.2f}s****')
            self.optim_params['lr'] = self.optimizer.param_groups[0]['lr']

            if epoch % 10 == 0:
                print(os.system('nvidia-smi'))
        return {"num_epochs":num_epochs, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc}
    
    def train_epoch(self, device, epoch=None,plot=False, max_iter=math.inf):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = []
        metric_logger = d_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', d_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        # metric_logger.add_meter('class_error', d_utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        with tqdm(self.train_dl, desc=f"Epoch {epoch}, Loss: ", unit="batch") as pbar:
            for i, (images, targets) in enumerate(pbar):
                images = images.to(device).tensors
                for t in targets:
                    for k, v in t.items():
                        if k == 'boxes':
                            t[k] = box_cxcywh_to_xyxy(v).to(device)
                        else:
                            t[k] = v.to(device)
                loss_dict, outputs = rcnn_forward(self.model, images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                self.model.eval()
                metric_logger.update(loss=loss_value)
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
                train_loss.append(loss_value)
                pbar.set_description(f"Epoch {epoch}, Loss: {loss_value:.4f}")
                if i >= max_iter:
                    break
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        print("Averaged stats:", stats)
        return stats

    def eval_epoch(self, device, epoch=None,plot=False, max_iter=math.inf):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        metric_logger = d_utils.MetricLogger(delimiter="  ")
        metric = MAP(iou_thresholds=[0.25, 0.5, 0.75])

        # metric_logger.add_meter('class_error', d_utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Test:'
        with tqdm(self.val_dl, desc=f"Epoch {epoch}, Loss: ", unit="batch") as pbar:
            for i, (images, targets) in enumerate(pbar):
                images = images.to(device).tensors
                for t in targets:
                    for k, v in t.items():
                        if k == 'boxes':
                            t[k] = box_cxcywh_to_xyxy(v).to(device)
                        else:
                            t[k] = v.to(device)
                nms_preds = []
                # print("targets", targets)
                with torch.no_grad():
                    # predictions = self.model(images)
                    loss_dict, predictions = rcnn_forward(self.model, images, targets)
                for i in range(len(predictions)):
                    p = predictions[i]
                    nms_indices = nms( p['boxes'],  p['scores'], self.nms_iou_thresh)
                    nms_boxes, nms_labels, nms_scores =  p['boxes'][nms_indices],  p['labels'][nms_indices], p['scores'][nms_indices]
                    nms_preds.append({'boxes':nms_boxes, 'labels':nms_labels, 'scores':nms_scores })
                metric.update(nms_preds, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                metric_logger.update(loss=loss_value)
                metric_logger.update(mAP=metric.compute()['map'])
                val_loss.append(loss_value)
                pbar.set_description(f"Epoch {epoch}, Loss: {loss_value:.4f}")
                if i >= max_iter:
                    break
            # acc = self.metric.compute()
                # val_acc += (diff[:,0] < (y[:,0]/10)).sum().item()  
            # print("number of val_acc: ", val_acc)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        print("Averaged stats:", stats)
        return stats

    def predict(self, model, test_dl, device, cls_names, plot_every=None, save_path=None):
        tot_targets = []
        tot_gt_targets = []
        with tqdm(test_dl, desc=f"Predict Loss: ", unit="batch") as pbar:
            for i, (images, targets) in enumerate(pbar):
                plot = (plot_every is not None) and (i % plot_every == 0)
                images = images.to(device).tensors
                for t in targets:
                    for k, v in t.items():
                        if k == 'boxes':
                            t[k] = box_cxcywh_to_xyxy(v).to(device)
                        else:
                            t[k] = v.to(device)
                batched_predictions = []
                tot_gt_targets.extend(targets)
                # print("targets", targets)
                with torch.no_grad():
                    loss_dict, predictions = rcnn_forward(model, images, targets)
                    boxes, labels = [], []
                    for p in predictions:
                        nms_indices = nms(p['boxes'], p['scores'], self.nms_iou_thresh)
                        nms_boxes, nms_labels, nms_scores = p['boxes'][nms_indices], p['labels'][nms_indices], p['scores'][nms_indices]
                        target = {'boxes':nms_boxes, 'labels':nms_labels, 'scores':nms_scores }
                        # tot_targets.append(target)
                        batched_predictions.append(target)
                tot_targets.extend(batched_predictions)
                if plot:
                    targets_np = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in targets]
                    pred_np = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in batched_predictions]
                    images_np = [im.detach().cpu().numpy() for im in images]
                    plot_images_with_bboxes(images_np, pred_np, cls_names, gt_target=targets_np, rescale=True, save_path=save_path,
                                             name=f"preds_{i}")
        return tot_targets, tot_gt_targets

# class DetrTrainer(Trainer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#     def train_epoch(self, device, epoch=None,plot=False, max_iter=math.inf):
#         loss_list = []
#         self.model.train()
#         self.criterion.train()
#         metric_logger = d_utils.MetricLogger(delimiter="  ")
#         metric_logger.add_meter('lr', d_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#         metric_logger.add_meter('class_error', d_utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#         header = 'Epoch: [{}]'.format(epoch)
#         print_freq = 50
#         # it = 0
#         # with tqdm(metric_logger.log_every(self.train_dl, print_freq, header), desc=f"Epoch {epoch}, Loss: ", unit="batch") as pbar:
#         for it, (samples, targets) in enumerate(metric_logger.log_every(self.train_dl, print_freq, header)):
#             samples = samples.to(device)
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#             labels = [t['labels'] for t in targets]
#             outputs = self.model(samples)
#             loss_dict = self.criterion(outputs, targets)
#             weight_dict = self.criterion.weight_dict
#             losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    
#             # reduce losses over all GPUs for logging purposes
#             loss_dict_reduced = d_utils.reduce_dict(loss_dict)
#             loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                             for k, v in loss_dict_reduced.items()}
#             loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                         for k, v in loss_dict_reduced.items() if k in weight_dict}
#             losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
    
#             loss_value = losses_reduced_scaled.item()
#             if not math.isfinite(loss_value):
#                 print("Loss is {}, stopping training".format(loss_value))
#                 print(loss_dict_reduced)
#                 sys.exit(1)

#             self.optimizer.zero_grad()
#             losses.backward()
#             if self.max_norm > 0:
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
#             self.optimizer.step()
#             loss_list.append(loss_value)
#             # pbar.set_description(f"Epoch {epoch}, Loss: {loss_value:.4f}")
#             metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#             metric_logger.update(class_error=loss_dict_reduced['class_error'])
#             metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

#             if it == max_iter:
#                 break
#         # gather the stats from all processes
#         metric_logger.synchronize_between_processes()
#         print("Averaged stats:", metric_logger)
#         stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#         return loss_list, stats 
        
#     @torch.no_grad()
#     def eval_epoch(self, device, epoch=None,plot=False, max_iter=math.inf):
#         self.model.eval()
#         self.criterion.eval()
#         loss_list = []
#         metric_logger = d_utils.MetricLogger(delimiter="  ")
#         metric_logger.add_meter('class_error', d_utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#         header = 'Test:'
    
#         iou_types = tuple(k for k in ('segm', 'bbox') if k in self.postprocessors.keys())
#         coco_evaluator = CocoEvaluator(self.base_ds, iou_types)
#         panoptic_evaluator = None
#         print_freq = 50
#         # with tqdm(metric_logger.log_every(self.val_dl, print_freq, header), desc=f"Epoch {epoch}, Loss: ", unit="batch") as pbar:
#         for it, (samples, targets) in enumerate(metric_logger.log_every(self.val_dl, print_freq, header)):
#             samples = samples.to(device)
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
#             outputs = self.model(samples)
#             loss_dict = self.criterion(outputs, targets)
#             weight_dict = self.criterion.weight_dict
    
#             # reduce losses over all GPUs for logging purposes
#             loss_dict_reduced = d_utils.reduce_dict(loss_dict)
#             loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                         for k, v in loss_dict_reduced.items() if k in weight_dict}
#             loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                             for k, v in loss_dict_reduced.items()}
#             metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                                     **loss_dict_reduced_scaled,
#                                     **loss_dict_reduced_unscaled)
#             metric_logger.update(class_error=loss_dict_reduced['class_error'])

#             loss_list.append(sum(loss_dict_reduced_scaled.values()).item())
    
#             orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#             # for i,(t, size) in enumerate(zip(targets, orig_target_sizes)):
#             #     if len(t['boxes']):
#             #         t['boxes'] = torch.stack([torch.tensor(convert_box_01_to_image_size(box_cxcywh_to_xyxy(box), size[1], size[0])) for box in t['boxes']])
#             #     targets[i] = t
#             results = self.postprocessors['bbox'](outputs, orig_target_sizes)
#             # acc = self.metric(results, targets) 
#             if 'segm' in self.postprocessors.keys():
#                 target_sizes = torch.stack([t["size"] for t in targets], dim=0)
#                 results = self.postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
#             res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#             if coco_evaluator is not None:
                
#                 coco_evaluator.update(res)
    
#             if panoptic_evaluator is not None:
#                 res_pano = self.postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
#                 for i, target in enumerate(targets):
#                     image_id = target["image_id"].item()
#                     file_name = f"{image_id:012d}.png"
#                     res_pano[i]["image_id"] = image_id
#                     res_pano[i]["file_name"] = file_name
    
#                 panoptic_evaluator.update(res_pano)

#             if it == max_iter:
#                 break
    
#         # gather the stats from all processes
#         metric_logger.synchronize_between_processes()
#         print("Averaged stats:", metric_logger)
#         if coco_evaluator is not None:
#             coco_evaluator.synchronize_between_processes()
#         if panoptic_evaluator is not None:
#             panoptic_evaluator.synchronize_between_processes()
    
#         # accumulate predictions from all images
#         if coco_evaluator is not None:
#             coco_evaluator.accumulate()
#             coco_evaluator.summarize()
#         panoptic_res = None
#         if panoptic_evaluator is not None:
#             panoptic_res = panoptic_evaluator.summarize()
#         stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#         if coco_evaluator is not None:
#             if 'bbox' in self.postprocessors.keys():
#                 stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#             if 'segm' in self.postprocessors.keys():
#                 stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
#         if panoptic_res is not None:
#             stats['PQ_all'] = panoptic_res["All"]
#             stats['PQ_th'] = panoptic_res["Things"]
#             stats['PQ_st'] = panoptic_res["Stuff"]
#         # print("stats:", stats)
#         stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#         # acc = self.metric.compute()
#         return loss_list, stats, coco_evaluator
    
    
#     @torch.no_grad()
#     def predict(self, model, test_dl, device, plot_every=None, save_path=None ):
#         model.eval()
#         tot_targets = []
#         tot_gt_targets = []
#         with tqdm(test_dl, desc=f"Predict Loss: ", unit="batch") as pbar:
#             for it, (samples, targets) in enumerate(pbar):
#                 plot = (plot_every is not None) and (it % plot_every == 0)
#                 samples = samples.to(device)
#                 targets = [{k: v.to(device) for k, v in t.items()} for t in targets]    
#                 outputs = model(samples)
#                 results = [{}]*len(targets)
#                 indices = self.criterion.matcher(outputs, targets)
#                 src_indices = self._get_src_permutation_idx(indices)
#                 results, targets = self.result_dict_from_logits(samples.tensors, targets, outputs['pred_boxes'], outputs["pred_logits"], src_indices[0], src_indices[1])
#                 # lens = [len(b) for b in boxes]
#                 # print(outputs['pred_boxes'].shape)
#                 # boxes = outputs['pred_boxes'][src_indices]
#                 # logits = outputs["pred_logits"][src_indices]
#                 # prob = F.softmax(logits, -1)
#                 # scores, labels = prob.max(-1)
#                 # results = [{'boxes': boxes[i], 'labels': labels[i], 'scores': scores[i]} for i in range(len(boxes))]
                
#                 # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#                 # print(len(orig_target_sizes))
#                 # results = self.postprocessors['bbox']({'pred_boxes': boxes, 'pred_logits': logits}, orig_target_sizes)
#                 tot_gt_targets.extend(targets)
#                 tot_targets.extend(results)
#                 if plot:
#                     targets_np = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in targets]
#                     pred_np = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in results]
#                     images_np = samples.tensors.detach().cpu().numpy()
#                     plot_images_with_bboxes(images_np, pred_np, gt_target=targets_np, save_path=save_path,
#                                              name=f"preds_{it}")
#         return tot_targets, tot_gt_targets 

#     def result_dict_from_logits(self, imgs, targets, proposal_boxes, proposal_logits, batch_indices, box_indices):       
#         chosen_boxes_list = []
    
#         for batch_idx in range(len(proposal_boxes)):
#             # Find matching indices for this batch_idx
#             matching_indices = np.where(batch_indices == batch_idx)[0]
        
#             # If there are matching indices, gather the chosen boxes, logits, and probabilities
#             if len(matching_indices) > 0:
#                 img_w, img_h = imgs[batch_idx].shape[1:]
#                 # print(img_w, img_h)
#                 chosen_boxes = proposal_boxes[batch_idx][box_indices[matching_indices]]
#                 chosen_boxes = torch.stack([torch.tensor(convert_box_01_to_image_size(box_cxcywh_to_xyxy(box), img_w, img_h)) for box in chosen_boxes])
#                 targets[batch_idx]['boxes'] = torch.stack([torch.tensor(convert_box_01_to_image_size(box_cxcywh_to_xyxy(box), img_w, img_h)) for box in targets[batch_idx]['boxes']])
#                 # chosen_boxes = convert_box_01_to_image_size(box_cxcywh_to_xyxy(chosen_boxes), img_w, img_h)
#                 logits = proposal_logits[batch_idx][box_indices[matching_indices]]
#                 prob = F.softmax(logits, -1)
#                 scores, labels = prob.max(-1)
#             else:
#                 # If no matching indices, create empty tensors
#                 chosen_boxes = torch.empty((0, 4), dtype=torch.float32)
#                 labels = torch.empty(0, dtype=torch.int64)
#                 scores = torch.empty(0, dtype=torch.float32)
            
#             # Create a dictionary for this batch_idx
#             batch_dict = {
#                 'boxes': chosen_boxes,
#                 'labels': labels,
#                 'scores': scores
#             }
        
#             # Append the dictionary to the chosen_boxes_list
#             chosen_boxes_list.append(batch_dict)
        
#         return chosen_boxes_list, targets


#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx
                
        

    

        
    

        
