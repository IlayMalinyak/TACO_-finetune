"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from .box_ops import rescale_bboxes

from pathlib import Path, PurePath

# COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#           [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_fit(
    fit_res: dict,
    fig=None,
    log_loss=False,
    legend=None,
    train_test_overlay: bool = False,
):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :param train_test_overlay: Whether to overlay train/test plots on the same axis.
    :return: The figure.
    """
    if fig is None:
        nrows = 1 if train_test_overlay else 2
        ncols = 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 5 * nrows),
            sharex="col",
            sharey=False,
            squeeze=False,
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(enumerate(["train", "val"]), enumerate(["loss", "acc"]))
    for (i, traintest), (j, lossacc) in p:

        ax = axes[j if train_test_overlay else i * 2 + j]

        attr = f"{traintest}_{lossacc}"
        data = fit_res[attr]
        label = traintest if train_test_overlay else legend
        h = ax.plot(np.arange(1, len(data) + 1), data, label=label)
        ax.set_title(attr)

        if lossacc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Accuracy (%)")

        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    return fig, axes

def plot_images_with_bboxes(images, target, cls_names, gt_target=None, rescale=False, ncols=2, save_path=None, name=None):
    """
    Plot a batch of images along with their corresponding bounding boxes and labels.

    Parameters:
        images (numpy.ndarray): Batch of images in the shape (batch_size, channels, height, width).
        bboxes (list of numpy.ndarray): List of bounding boxes for each image in the batch.
        labels (list of numpy.ndarray): List of labels for each bounding box.
        class_names (list): List of class names for the labels.
        ncols (int, optional): Number of columns in the subplot grid. Defaults to 2.
    """


    bboxes = [t['boxes'] for t in target]
    labels = [t['labels'] for t in target]
    scores = [t['scores'] for t in target]
    
    assert len(images) == len(bboxes) == len(labels), "Number of images, bounding boxes, and labels must be the same."

    if gt_target is not None:
        gt_bboxes = [t['boxes'] for t in gt_target]
        gt_labels = [t['labels'] for t in gt_target]
       
    batch_size = len(images)
    nrows = max(batch_size // ncols, 1)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(36,18))
    plt.subplots_adjust(hspace=0.2)

    for i in range(batch_size):
        ax = axes[i // ncols, i % ncols] if nrows > 1 else axes[i]

        # Transpose the image to (height, width, channels) for displaying.
        img = np.transpose(images[i], (1, 2, 0))
        img = (img - img.min())/(img.max()-img.min())
        # img = images[i]

        # Plot the image
        ax.imshow(img)
        ax.axis('off')

        # Plot bounding boxes and labels
        for bbox, label, score in zip(bboxes[i], labels[i], scores[i]):
            # print('before resize', bbox)
            # print('image shape', img.shape)
            # if max((bbox[0], bbox[1])) < 1:  
            # bbox = torch.tensor(bbox)         
            # bbox = convert_box_01_to_image_size(box_cxcywh_to_xyxy(torch.tensor(bbox)), img.shape[1], img.shape[0])
            # print("pred:", bbox)
            x_min, y_min, x_max, y_max = bbox
            if rescale:
                x_min, x_max = int(x_min*img.shape[1]), int(x_max*img.shape[1])
                y_min, y_max = int(y_min*img.shape[0]), int(y_max*img.shape[0])
           
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 fill=False, edgecolor='r', linewidth=2)
            ax.add_patch(rect)
            text = f'{cls_names[label]}: {score:.2f}'
            ax.text(x_min, y_min,text, bbox=dict(facecolor='yellow', alpha=0.7, pad=0.3),
                    fontsize=32, color='black')
        if gt_target is not None:
            for bbox, label in zip(gt_bboxes[i], gt_labels[i]):
                # print("gt:", bbox)
                # bbox = torch.tensor(bbox)
                # if max((bbox[0], bbox[1])) < 1:           
                # bbox = convert_box_01_to_image_size(box_cxcywh_to_xyxy(torch.tensor(bbox)), img.shape[1], img.shape[0])
                x_min, y_min, x_max, y_max = bbox
                if rescale:
                    x_min, x_max = int(x_min*img.shape[1]), int(x_max*img.shape[1])
                    y_min, y_max = int(y_min*img.shape[0]), int(y_max*img.shape[0])
                \
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     fill=False, edgecolor='b', linewidth=2)
                ax.add_patch(rect)
                ax.text(x_min, y_max, cls_names[label], bbox=dict(facecolor='white', alpha=0.7, pad=0.3),
                        fontsize=32, color='black')
    if save_path is not None:
        print("saving plot at ", f'{save_path}/{name}.png')
        plt.savefig(f'{save_path}/{name}.png')
        plt.clf()
    else:
        # plt.clf()
        plt.show()

def plot_attention_map(images, outputs, conv_features, cls_names, save_path=None, name=None):

    for i in range(len(outputs)):
        bboxes = outputs[i]['boxes']
        labels = outputs[i]['labels']
        scores = outputs[i]['scores']
        dec_attn_weights = outputs[i]['dec_attn']
        
        image = np.transpose(images[i], (1, 2, 0))
        image = (image - image.min())/(image.max()-image.min())
        
        h, w = conv_features['0'].tensors.shape[-2:]
        fig, axs = plt.subplots(ncols=len(bboxes), nrows=2)
        for idx, (ax_i, (xmin, ymin, xmax, ymax)) in enumerate(zip(axs.T, bboxes)):
            ax = ax_i[0] if len(bboxes) > 1 else axs[0]
            ax.imshow(dec_attn_weights[idx].reshape((h, w)))
            ax.axis('off')
            ax.set_title(f'decoder attention weights')
            ax = ax_i[1] if len(bboxes) > 1 else axs[1]
            ax.imshow(image)
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color='blue', linewidth=2))
            ax.axis('off')
            # text = f'{cls_names[labels[idx]]}: {scores[idx]:.2f}'
            ax.set_title('object')
        fig.tight_layout()
        if save_path is not None:
            print("saving plot at ", f'{save_path}/{name}_{i}_attn.png')
            plt.savefig(f'{save_path}/{name}_{i}_attn.png')
            plt.clf()
        else:
            plt.show()


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt', train_test=True):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]
    
    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        # for c in df.columns:
        #     print(c, df[c].dtype)
        # print(df.columns)
        for j, field in enumerate(fields):
            if field == 'coco_mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            if field == 'mAP':
                try:
                    df_no_coco = df.drop(columns=['test_coco_eval_bbox'])
                except KeyError:
                    print("no coco eval in log file")
                    df_no_coco = df
                ax = df_no_coco.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-'],
                    label=['validation mAP']
                )
                # ax.set_xticklabels(len(df[f'test_{field}']))
            else:
                if train_test:
                    try:
                        df_no_coco = df.drop(columns=['test_coco_eval_bbox'])
                    except KeyError:
                        print("no coco eval in log file")
                        df_no_coco = df
                    cur_ax = axs[j] if len(fields) > 1 else axs
                    ax = df_no_coco.interpolate().ewm(com=ewm_col).mean().plot(
                        y=[f'train_{field}', f'test_{field}'],
                        ax=cur_ax,
                        color=[color] * 2,
                        style=['-', '--'],
                        label=['train', 'validation']
                    )
                else:
                    try:
                        df_no_coco = df.drop(columns=['test_coco_eval_bbox'])
                    except KeyError:
                        print("no coco eval in log file")
                        df_no_coco = df
                    cur_ax = axs[j] if len(fields) > 1 else axs
                    ax = df_no_coco.interpolate().ewm(com=ewm_col).mean().plot(
                        y=[f'train_{field}'],
                        ax=cur_ax,
                        color=[color],
                        style=['-'],
                        label=['train']
                    )

                # ax.set_xticklabels(len(df[f'train_{field}']))
    if len(fields) > 1:
        for ax, field in zip(axs, fields):
            # ax.legend([Path(p).name for p in logs])
            ax.legend()
            ax.set_title(field)
            ax.set_xlabel('Epoch')
    else:
        axs.legend()
        axs.set_title(fields[0])
        axs.set_xlabel('Epoch')


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs
