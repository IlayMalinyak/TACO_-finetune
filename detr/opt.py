from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import optuna
import random

alpha_freqs = [0,0.01, 0.36, 0.023, 0.05, 0.105, 1, 0.016, 0.0005]


def objective(trial):
    # lr = trial.suggest_float('lr', 1e-5, 1e-3)
    alpha_1 = trial.suggest_categorical('alpha_1', [0.6,0.7,0.8,0.9,1])
    alpha_3 = trial.suggest_categorical('alpha_3', [0.6,0.7,0.8,0.9,1])
    alpha_6 = trial.suggest_categorical('alpha_6', [0.8,0.9,1])
    use_freqs = trial.suggest_categorical('use_freqs', [True, False])
    use_focal = trial.suggest_categorical('use_focal', [True, False])
    alpha_t_list = [0, alpha_1, 1, alpha_3, 1, 1, alpha_6, 1, 0.0005] if not use_freqs else alpha_freqs
    gamma = trial.suggest_float('gamma', 1, 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = utils.Cfg(device=device, batch_size=1, lr=1e-4, coco_path='/home/ilay.kamai/mini_project/data',
                        resume="detr/detr-r50_no-class-head.pth",
                        output_dir='/home/ilay.kamai/mini_project/detr/optuna_output',
                        weight_decay=8e-4, early_stopping=10, alpha_t=alpha_t_list, gamma=gamma,
                        focal_loss=use_focal,
                        
                        )

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, criterion, postprocessors = build_model(args)
    for param in model.backbone.parameters():
        param.requires_grad = False
    # for param in model.transformer.parameters():
    #     param.requires_grad = False
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    
    base_ds = get_coco_api_from_dataset(dataset_val)


    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    output_dir = Path(args.output_dir)
    for epoch in range(4):
        os.system("nvidia-smi")
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        val_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
        val_loss = val_stats['loss']
        val_acc = val_stats['mAP']
        trial.report(val_acc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc

if __name__ == "__main__":
    study = optuna.create_study(study_name='opt_detr_max', load_if_exists=True, direction='maximize')
    study.optimize(objective, n_trials=50, gc_after_trial=True)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    print(study.best_params)
                        
        

        
