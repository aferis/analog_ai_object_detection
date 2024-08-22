"""
Train a model on a custom dataset

Usage:
    -> should be run directly
    e.g.: $ train.py --save-weights True --use-analog True --img-size 480 --batch-size 8 --epochs 400
"""

# Default libraries & packages
import os
import csv
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda import amp  # Automatic mixed precision package -> https://pytorch.org/docs/stable/amp.html
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import logging
import math
import random
import sys
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from threading import Thread
from collections import OrderedDict

# Inside analog_ai_object_detection
from global_settings import import_config_dict
import val
from utils.plots import plot_labels, plot_images, plot_results
from utils.torch_utils import EarlyStopping, de_parallel, time_sync
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, strip_optimizer, str2bool, quantize_standard
from utils.loss import ComputeLoss
from models.yolov5.yolo import Model
from rpu_settings import rpu_config, tune_rpu_hyperpamaters

# AIHWKIT Imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.nn.modules.conv import *
from aihwkit.simulator.configs.configs import *
from aihwkit.simulator.configs.devices import *

# Initialize Logger
LOGGER = logging.getLogger("MainLogger")
LOGGER.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s %(module)s:%(lineno)d] %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # analog_ai_object_detection directory

# Add ROOT to PATH
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def train(args,
          config_dict,
          device,
          USE_CUDA):

    cluster, noval, save_weights, use_analog, hybrid, seed_count, q_min, q_max, p_backbone, p_head, input_img_size, \
        batch_size, batch_size_test, epochs = args.cluster, args.noval, args.save_weights, args.use_analog, \
        args.hybrid, args.seed_count, args.quantize_min, args.quantize_max, args.precision_backbone, \
        args.precision_head,  args.img_size, args.batch_size, args.batch_size_test, args.epochs

    # Load the dataset configuration file as a dict
    dataset_config_file = os.path.join(ROOT, config_dict['DATASET']['config_dataset'])
    if isinstance(dataset_config_file, (str, Path)):
        with open(dataset_config_file, mode="r") as f:
            dataset_config_dict = yaml.safe_load(f)

    # Check if the necessary elements exist in the dataset config file
    assert 'train' or 'train_local' or 'train_cluster' in dataset_config_dict, "'train' key is missing"
    assert 'val' or 'val_local' or 'val_cluster' in dataset_config_dict, "'val' key is missing"
    assert 'test' or 'test_local' or 'test_cluster' in dataset_config_dict, "'test' key is missing"
    assert 'nc' in dataset_config_dict, "'nc' key (number of classes) is missing."
    if 'names' not in dataset_config_dict:
        dataset_config_dict['names'] = [f'class{i}' for i in range(dataset_config_dict['nc'])]

    num_classes, class_names = dataset_config_dict['nc'], dataset_config_dict['names']
    assert len(class_names) == num_classes, f'{len(class_names)} class names were found for nc={num_classes}'

    ## Path distinction (local or cluster)
    if cluster:
        save_dir_train = config_dict['SAVEPATH']['save_dir_train_cluster']
        train_path = dataset_config_dict['train_cluster']
        val_path = dataset_config_dict['val_cluster']
        test_path = dataset_config_dict['test_cluster']
        path_split = 'repos/'
    else:
        save_dir_train = config_dict['SAVEPATH']['save_dir_train_local']
        train_path = dataset_config_dict['train_local']
        val_path = dataset_config_dict['val_local']
        test_path = dataset_config_dict['test_local']
        path_split = 'Datasets/'

    # Directories
    ## Create a unique directory for each training run
    architecture = config_dict['NETWORK']['architecture']
    timestr = time.strftime("%Y.%m.%d-%H:%M:%S")
    pid = os.getpid()
    lr = config_dict['HYPERPARAMETER']['learning_rate']
    if use_analog:
        analog_label = config_dict['RPU']['configuration'].title().replace(" ", "")
        label_nn = f"A_{analog_label}"
        save_dir = save_dir_train + f"{label_nn}_{architecture[-1]}_{timestr}_{input_img_size}_" \
                                    f"bs{batch_size}_lr{lr}_e{epochs}-{pid}"
    else:
        label_nn = f"D_{p_backbone}BB_{p_head}Head"
        save_dir = save_dir_train + f"{label_nn}_{architecture[-1]}_{timestr}_{input_img_size}_" \
                                    f"bs{batch_size}_lr{lr}_e{epochs}-{pid}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ## Save all configurations
    with open(save_dir / 'dataset_config.yaml', 'w') as f:
        yaml.safe_dump(dataset_config_dict, f, sort_keys=False)
    with open(save_dir / 'image.yaml', 'w') as f:
        yaml.safe_dump(config_dict['IMAGE'], f, sort_keys=False)
    with open(save_dir / 'network.yaml', 'w') as f:
        yaml.safe_dump(config_dict['NETWORK'], f, sort_keys=False)
    with open(save_dir / 'optimizer.yaml', 'w') as f:
        yaml.safe_dump(config_dict['OPTIMIZER'], f, sort_keys=False)
    with open(save_dir / 'rpu.yaml', 'w') as f:
        yaml.safe_dump(config_dict['RPU'], f, sort_keys=False)
    with open(save_dir / 'scheduler.yaml', 'w') as f:
        yaml.safe_dump(config_dict['SCHEDULER'], f, sort_keys=False)
    with open(save_dir / 'stopper.yaml', 'w') as f:
        yaml.safe_dump(config_dict['STOPPER'], f, sort_keys=False)

    ## Paths for saving last and the best training weights
    save_dir_weights = save_dir / 'weights'
    save_dir_weights.mkdir(parents=True, exist_ok=True) # Make dir
    last, best = save_dir_weights / 'last.pt', save_dir_weights / 'best.pt'

    # Create a log file
    log_file = save_dir / 'log.txt'
    fh = logging.FileHandler(filename=log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)

    # Input arguments
    args = vars(args)
    with open(save_dir / 'args.yaml', 'w') as f:
        yaml.safe_dump(args, f, sort_keys=False)

    # Hyperparameters
    hyp = config_dict['HYPERPARAMETER']
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    use_stopper = config_dict['STOPPER'].get('use_stopper', False)

    # Random Seeds
    LOGGER.info(f"Training & validation using {seed_count} different seeds -> results are then averaged in the end")
    df_results_dict = OrderedDict() # Dict containing results df of each seed
    for seed in range(1, 1 + seed_count):
        LOGGER.info(f"Seed: {seed}/{seed_count}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Setup RPU configuration
        RPU_config = config_dict['RPU']['configuration']
        RPU_params = config_dict['RPU']['value']
        device_config = rpu_config(RPU_config, RPU_params)

        # Modifying RPU parameters
        tune_rpu_param = config_dict['RPU']['tune_rpu_param']
        if tune_rpu_param and isinstance(device_config.device, IdealDevice): # Specify only seed for IdealDevice
            device_config.device.construction_seed = seed
        elif tune_rpu_param and 'preset' in RPU_config.lower(): # Specify only seed & is_perfect for presets
            device_config.device.construction_seed = seed
            device_config.forward.is_perfect = config_dict['RPU']['fw_is_perfect']
            device_config.backward.is_perfect = config_dict['RPU']['bw_is_perfect']
        elif tune_rpu_param and not isinstance(device_config, FloatingPointRPUConfig):
            tune_rpu_hyperpamaters(device_config, config_dict['RPU'], seed)

        # Get YOLO network architecture
        model_config_file = os.path.join(ROOT, 'models/yolov5/', architecture) + '.yaml'
        if isinstance(model_config_file, (str, Path)):
            with open(model_config_file, mode="r") as f:
                model_config_dict = yaml.safe_load(f)

        # Length of backbone and head
        num_bb_modules = len(model_config_dict['backbone']) # Number of backbone modules
        num_head_modules = len(model_config_dict['head'])   # Number of head modules

        # Network model
        model = Model(cfg=model_config_dict, ch=3, nc=num_classes, anchors=hyp.get('anchors'))
        if use_analog:  # analog NN
            if seed == 1:
                LOGGER.info('Using analog NN')
                LOGGER.info(f'Canonical device configuration: {repr(device_config)}')
            model_parameters_digital = list(model.parameters())
            model = convert_to_analog(model, device_config, weight_scaling_omega=0.6)
            model_parameters_analog = list(model.parameters())
            model.to(device)

        else:   # digital NN
            if seed == 1:
                LOGGER.info('Using digital NN')
            model.to(device)

        # Optimizer
        opt = config_dict['OPTIMIZER']['optimizer']
        lr = hyp.get('learning_rate', 0.1)
        momentum = hyp.get('momentum', 0.9)
        weight_decay = hyp.get('weight_decay', 1e-4)

        if use_analog:
            if opt.lower() == 'adam':
                LOGGER.info(f"ADAM is not compatible for analog AI, using SGD instead.")
                opt = config_dict['OPTIMIZER']['optimizer'] = 'sgd'
            optimizer = AnalogSGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            optimizer.regroup_param_groups(model)

        else:
            g0 = []  # List containing weights of all BatchNorm2d layers
            g1 = []  # List containing weights of all Conv2d layers
            g2 = []  # List containing bias of all BatchNorm2d layers & 3 Conv2d layers in Detect()

            ## Iterate through all modules & layers for 3 unique optimizer parameter groups
            for v in model.modules():
                if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # Bias
                    g2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d):  # Weight (no decay)
                    g0.append(v.weight)
                elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # Weight (with decay)
                    g1.append(v.weight)
            if opt.lower() == 'adam':
                optimizer = Adam(g0, lr=lr, betas=(momentum, 0.999))  # Adjust beta1 to momentum
            elif opt.lower() == 'sgd':
                optimizer = SGD(g0, lr=lr, momentum=momentum, nesterov=True)
            optimizer.add_param_group({'params': g1, 'weight_decay': weight_decay})  # Add g1 with weight_decay
            optimizer.add_param_group({'params': g2})  # Add g2 (biases)
            LOGGER.info(f"Optimizer: {type(optimizer).__name__} with parameter groups {len(g0)} weight, "
                        f"{len(g1)} weight (no decay), and {len(g2)} bias")
            del g0, g1, g2

        # Scheduler
        if config_dict['SCHEDULER']['use_scheduler']:
            lrf = config_dict['SCHEDULER'].get('lrf', 0.2)
            if config_dict['SCHEDULER'].get('lr_lambda') == 'linear':
                lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - lrf) + lrf   # linear
                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
            elif config_dict['SCHEDULER'].get('lr_lambda') == 'nonlinear':
                # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
                y1 = 1
                y2 = lrf
                lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (y2 - y1) + y1
                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
            else:
                scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config_dict['SCHEDULER']['milestones'],
                                                     gamma=config_dict['SCHEDULER']['gamma'])

        # Image sizes
        img_size = input_img_size
        grid_size = max(int(model.stride.max()), 32)
        num_detection_layers = model.model[-1].num_detect_layers    # Used for scaling hyp['obj']

        ## Verify image size is a multiple of stride s in each dimension
        if isinstance(input_img_size, int):
            img_size = max(math.ceil(input_img_size / grid_size) * grid_size, grid_size * 2)
        elif isinstance(input_img_size, list):
            img_size = [max(math.ceil(img_size / grid_size) * grid_size, grid_size * 2) for x in img_size]
        else:
            raise TypeError("'img_size' must be an integer or a list!")

        if img_size != input_img_size:
            print(f'WARNING: Image size must be multiple of max stride {grid_size} -> '
                  f'Updating img_size from {input_img_size} to {img_size}')

        # TrainLoader
        data_augmentation = config_dict['IMAGE'].get('augment', True)
        workers = config_dict['IMAGE']['dataload_workers']
        train_loader, dataset = create_dataloader(path=train_path, imgsz=img_size, batch_size=batch_size,
                                                  stride=grid_size, hyp=hyp, augment=data_augmentation,
                                                  workers=workers, prefix='Train: ')
        max_lc = int(np.concatenate(dataset.labels, 0)[:, 0].max())     # Max label class
        num_batches = len(train_loader)
        assert max_lc < num_classes, f'Label class {max_lc} exceeds nc={num_classes} in {dataset_config_file}. ' \
                                     f'Possible class labels are 0-{num_classes - 1}'
        labels = np.concatenate(dataset.labels, 0)
        plot_labels(labels, class_names, save_dir)

        # ValidationLoader
        val_loader = create_dataloader(path=val_path, imgsz=img_size, batch_size=batch_size,
                                       stride=grid_size, hyp=hyp, augment=False, rect=True,
                                       workers=workers, pad=0.5, prefix='Val: ')[0]

        # Model parameters
        hyp['box_loss_gain'] *= 3. / num_detection_layers                         # Scale to layers
        hyp['cls_loss_gain'] *= num_classes / 80. * 3. / num_detection_layers     # Scale to classes and layers
        hyp['obj_loss_gain'] *= (img_size / 640) ** 2 * 3. / num_detection_layers # Scale to image size and layers
        model.num_classes = num_classes                                           # Attach num_classes to model
        model.hyp = hyp                                                           # Attach hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, num_classes).to(device) * num_classes  # Attach class weights
        model.names = class_names

        # Tensorboard
        if seed == 1:
            tb_writer = SummaryWriter(save_dir)
            for k, v in hyp.items():
                tb_writer.add_text(k, str(v), 0)
            tb_writer.add_text('network', architecture, 0)
            tb_writer.add_text('train_path', train_path.rsplit(path_split, 1)[1], 0)
            tb_writer.add_text('val_path', val_path.rsplit(path_split, 1)[1], 0)
            tb_writer.add_text('RPU_config', RPU_config, 0)
            tb_writer.add_text('data_augmentation', str(data_augmentation), 0)
            tb_writer.add_text('optimizer', opt, 0)
            tb_writer.add_text('dataload_workers', str(workers), 0)

        # Start training
        start_epoch = 0
        if seed == 1:
            best_fitness = 0.0
        t0 = time.time()
        '''(For Warmup) num_warmup_iter = max(round(hyp['warmup_epochs'] * num_batches), 1000) '''
        last_opt_step = -1
        mAPs = np.zeros(num_classes)    # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0) # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        if config_dict['SCHEDULER']['use_scheduler']:
            scheduler.last_epoch = start_epoch - 1  # (Do not move)
        scaler = amp.GradScaler(enabled=USE_CUDA)   # To prevent underflow
        if use_stopper:
            stopper = EarlyStopping(patience=config_dict['STOPPER']['patience'])
        compute_loss = ComputeLoss(model)   # Init loss class
        if seed == 1:
            LOGGER.info(f'\nImage sizes: {img_size} train, {img_size} val\n'
                        f'Using {train_loader.num_workers} dataloader workers\n'
                        f"Logging results to {save_dir}\n"
                        f'Starting training for {epochs} epochs...')

        '''
        # --> Check weight intervals (1)
        l_min = []
        l_max = []
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm2d) and hasattr(m, 'weight'):
                # m.analog_tile.rpu_config
                l_min.append(m.weight.min())
                l_max.append(m.weight.max())

        # if min(l_min) < -0.3:
        print(f"\nMin: {min(l_min)}, size: {len(l_min)}")
        # if max(l_max) > 0.3:
        print(f"Max: {max(l_max)}, size: {len(l_max)}")
        '''
        for epoch in range(epochs):
            # Discretization of weights and biases considering backbone- & head-precisions (only for digital NN)
            # tt1 = time_sync()
            if not use_analog and (p_backbone <= 8 or p_head <= 8 or (q_min and q_max)):  # Precision > 8 is too slow!
                for i, m in enumerate(model.model):
                    if i < num_bb_modules and (p_backbone <= 8 or (q_min and q_max)):
                        for name, param in m.named_parameters():
                            if param.requires_grad:
                                param.data.copy_(quantize_standard(param, min=q_min, max=q_max,
                                                                   device=param.device, num_bits=p_backbone))
                    elif num_bb_modules <= i <= (num_bb_modules + num_head_modules) and (p_head <= 8 or (q_min and q_max)):
                        for name, param in m.named_parameters():
                            if param.requires_grad:
                                param.data.copy_(quantize_standard(param, min=q_min, max=q_max,
                                                                   device=param.device, num_bits=p_head))
            # tt2 = time_sync()
            # print(f'\n{tt2-tt1}\n')

            model.train()
            mean_loss = torch.zeros(3, device=device)
            progress_bar = enumerate(train_loader)
            LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
            progress_bar = tqdm(progress_bar, total=num_batches)
            optimizer.zero_grad()

            for i, (imgs, targets, paths, _) in progress_bar:   # batch
                """
                i: idx
                imgs: shape(batch_size, num_channels (3), resolution_x, resolution_y)
                targets: Tensor shape(object, num_attributes) with attributes = idx_img (0..batch_size), class, x, y, w, h
                paths: List containing image paths (size: batch_size)
                """
                # ---------------------------------- Start batch -------------------------------------------------------
                num_integr_batch = i + num_batches * epoch                 # Number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                '''
                # --> Warmup steps in the original framework
                if not use_analog and num_integr_batch <= num_warmup_iter:
                    x_interp = [0, num_warmup_iter]
                    nom_batch_size = 64
                    accumulate = max(1, np.interp(num_integr_batch, x_interp, [1, nom_batch_size / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        if config_dict['SCHEDULER']['use_scheduler'] and config_dict['SCHEDULER']['lr_lambda'] != 'multistep':
                            x['lr'] = np.interp(num_integr_batch, x_interp, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        else:
                            x['lr'] = np.interp(num_integr_batch, x_interp, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['lr']])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(num_integr_batch, x_interp, [hyp['warmup_momentum'], hyp['momentum']])
                '''

                # Forward
                if use_analog:
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))
                else:
                    with amp.autocast(enabled=USE_CUDA):    # Forward inference in mixed precision (float32 & float16)
                        pred = model(imgs)  # forward
                        loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

                '''
                # --> Original digital step with scaler
                if not use_analog:   
                    # Backward
                    scaler.scale(loss).backward()
                    # Optimize
                    if num_integr_batch - last_opt_step >= accumulate:
                        scaler.step(optimizer)  # optimizer.step
                        scaler.update()
                        optimizer.zero_grad()
                        last_opt_step = num_integr_batch
                
                else:
                '''
                # Backward and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Log
                mean_loss = (mean_loss * i + loss_items) / (i + 1)   # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g} GB'  # (GB)
                progress_bar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mean_loss, targets.shape[0], imgs.shape[-1]))

                # Plot example training batch images
                if num_integr_batch < 3:
                    p = save_dir / f'train_batch{num_integr_batch}.jpg' # File name
                    Thread(target=plot_images, args=(imgs, targets, paths, p, class_names), daemon=True).start()

            # ---------------------------------- End batch -------------------------------------------------------------

            '''
            # --> Check weight intervals (2)
            g2_min, g2_max, g0_min, g0_max, g1_min, g1_max = [], [], [], [], [], []
            for m in model.modules():
                if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):  # Bias
                    g2_min.append(m.bias.min()) if isinstance(m.bias.min(), float) else g2_min.append(float(m.bias.min()))
                    g2_max.append(m.bias.max()) if isinstance(m.bias.max(), float) else g2_max.append(float(m.bias.max()))
                if isinstance(m, nn.BatchNorm2d):  # Weight (no decay)
                    g0_min.append(m.weight.min()) if isinstance(m.weight.min(), float) else g0_min.append(float(m.weight.min()))
                    g0_max.append(m.weight.max()) if isinstance(m.weight.max(), float) else g0_max.append(float(m.weight.max()))
                elif hasattr(m, 'weight'): # and isinstance(v.weight, nn.Parameter):  # Weight (with decay)
                    g1_min.append(m.weight.min()) if isinstance(m.weight.min(), float) else g1_min.append(float(m.weight.min()))
                    g1_max.append(m.weight.max()) if isinstance(m.weight.max(), float) else g1_max.append(float(m.weight.max()))

            #if min(l_min) < -0.3:
            print(f"\n g2: Min: {min(g2_min)}, Max: {max(g2_max)} size_min: {len(g2_min)}, size_max: {len(g2_max)}")
            print(f"\n g0: Min: {min(g0_min)}, Max: {max(g0_max)} size_min: {len(g0_min)}, size_max: {len(g0_max)}")
            print(f"\n g1: Min: {min(g1_min)}, Max: {max(g1_max)} size_min: {len(g1_min)}, size_max: {len(g1_max)}")
            #if max(l_max) > 0.3:
            '''

            # Scheduler
            if use_analog:
                lr_param_groups = optimizer.param_groups[0]['lr']
            else:
                lr_param_groups = [x['lr'] for x in optimizer.param_groups]  # for loggers
            if config_dict['SCHEDULER']['use_scheduler']:
                scheduler.step()

            # mAP
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop if use_stopper else (epoch + 1 == epochs)

            # Calculate mAP
            if not noval or final_epoch:
                # results: [P, R, mAP@.5, mAP@.5-.95, box loss, object loss, class loss]
                # mAPs: Array of mAPs (per class)
                results, mAPs, _ = val.run(data=dataset_config_dict,
                                           seed=seed,
                                           seed_count=seed_count,
                                           batch_size=batch_size,
                                           imgsz=img_size,
                                           conf_thres=hyp.get('conf_t', 0.001),
                                           iou_thres=hyp.get('iou_t', 0.6),
                                           device=device,
                                           model=model,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=final_epoch,
                                           compute_loss=compute_loss,
                                           config_dict=config_dict,
                                           USE_CUDA=USE_CUDA,
                                           cluster=cluster)

            # Update best mAP
            results_arr = np.array(results).reshape(1, -1)
            w_results4 = [0.0, 0.0, 0.1, 0.9]               # Weights for [P, R, mAP@0.5, mAP@0.5:0.95]
            fi = (results_arr[:, :4] * w_results4).sum(1)   # Weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi

            # Save and plot training-/validation results, model evaluation metrics and parameters
            if use_analog:
                log_vals = list(mean_loss) + list(results) + [lr_param_groups] + [float(mem[:-3])]
                log_keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',                           # Train loss
                         'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # Metrics
                         'val/box_loss', 'val/obj_loss', 'val/cls_loss',                                    # Val loss
                         'x/lr', 'memory [GB]']                                                             # Single lr & GPU memory
            else:
                log_vals = list(mean_loss) + list(results) + lr_param_groups + [float(mem[:-3])]
                log_keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',                           # Train loss
                         'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # Metrics
                         'val/box_loss', 'val/obj_loss', 'val/cls_loss',                                    # Val loss
                         'x/lr0', 'x/lr1', 'x/lr2', 'memory [GB]']                                          # lr's & GPU memory
            results_dict = {k: v for k, v in zip(log_keys, log_vals)}
            if seed_count > 1:
                save_dir_results = save_dir / f'results{seed}.csv'
            else:
                save_dir_results = save_dir / f'results.csv'
            results_num_columns = len(results_dict) + 1
            results_header = '' if save_dir_results.exists() else \
                (('%20s,' * results_num_columns % tuple(['epoch'] + log_keys)).rstrip(',') + '\n')
            with open(save_dir_results, 'a') as f:
                f.write(results_header + ('%20.5g,' * results_num_columns % tuple([epoch] + log_vals)).rstrip(',') + '\n')
            if seed == 1:
                for k, v in results_dict.items():
                    tb_writer.add_scalar(k, v, epoch)
            if seed_count > 1:
                df_results_dict[seed] = pd.read_csv(save_dir_results)

            # Save last, best and delete
            if save_weights:
                # Save model
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': model if use_analog else deepcopy(model).half(),
                        'optimizer': optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt

            # Stop Single-GPU
            if use_stopper and stopper(epoch=epoch, fitness=fi):
                break

        # ---------------------------------- End epoch -----------------------------------------------------------------
        # ---------------------------------- End training --------------------------------------------------------------

        LOGGER.info(f'Seed {seed}: {epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')

    # Take cell average of all results.csv files (of each seed)
    if seed_count > 1:
        df_results_dict_filtered = df_results_dict.copy()
        num_df_results = len(df_results_dict_filtered)
        df_results_sum = df_results_dict_filtered.pop(list(df_results_dict_filtered.keys())[0])
        for seed_key in df_results_dict_filtered:
            df_results_sum += df_results_dict_filtered[seed_key]
        df_results_avg = df_results_sum / num_df_results
        df_results_avg.to_csv(path_or_buf=save_dir / 'results.csv', index=False, mode='a')

    # Plot results
    if seed_count > 1:
        for i in range(1, 1+ num_df_results):
            plot_results(file=save_dir / f'results{i}.csv')
    plot_results(file=save_dir / 'results.csv')
    files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
    files = [(save_dir / file) for file in files if (save_dir / file).exists()]  # filter
    for f in files:
        tb_writer.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats='HWC')
    LOGGER.info(f"Results saved to {save_dir}")

    # Test
    save_dir_test = Path(os.path.join(save_dir, 'test/'))
    if save_weights:
        for ckpt in last, best:
            if ckpt.exists():
                strip_optimizer(ckpt)  # Strip optimizers
                if ckpt is best:
                    LOGGER.info(f'\nTesting {ckpt}...')
                    save_dir_test.mkdir(parents=True, exist_ok=True)
                    results_test, _, t_test = val.run(data=dataset_config_dict,
                                                      weights=ckpt,
                                                      batch_size=batch_size_test,
                                                      imgsz=img_size,
                                                      conf_thres=0.25,
                                                      iou_thres=0.45,
                                                      device=device,
                                                      analog=hybrid and not use_analog,
                                                      dataloader=None,
                                                      save_dir=save_dir_test,
                                                      plots=final_epoch,
                                                      compute_loss=compute_loss,
                                                      config_dict=config_dict,
                                                      USE_CUDA=USE_CUDA,
                                                      cluster=cluster)

                    test_loss_keys = ['test/precision', 'test/recall', 'test/mAP_0.5', 'test/mAP_0.5:0.95',
                                      'test/box_loss', 'test/obj_loss', 'test/cls_loss',
                                      't_total[s]/pre-process', 't_total[s]/inference', 't_total[s]/NMS',
                                      't_per_img[ms]/pre-process', 't_per_img[ms]/inference', 't_per_img[ms]/NMS']
                    test_loss_dict = {k: v for k, v in zip(test_loss_keys, list(results_test) + list(t_test))}
                    with open(save_dir_test / 'test_loss.csv', 'w') as f:
                        f.write((('%20s,' * len(test_loss_dict) % tuple(test_loss_keys)).rstrip(',') + '\n') +
                                ('%20.5g,' * len(test_loss_dict) % tuple(list(results_test) + list(t_test))).rstrip(','))

    torch.cuda.empty_cache()
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=str2bool, default=False, help='Whether to use GPU cluster')
    parser.add_argument('--noval', type=str2bool, default=False, help='Only validate the final epoch')
    parser.add_argument('--save-weights', type=str2bool, default=True, help='Whether to save weights')
    parser.add_argument('--use-analog', type=str2bool, default=False, help='Whether to use analog')
    parser.add_argument('--hybrid', type=str2bool, default=False, help='Whether to combine analog & digital for inference')
    parser.add_argument('--seed-count', type=int, default=1, help='Total number of seed counts')
    parser.add_argument('--quantize-min', type=float, default=None, help='Min. value for quantization')
    parser.add_argument('--quantize-max', type=float, default=None, help='Max. value for quantization')
    parser.add_argument('--precision-backbone', type=int, default=16, help='Bits to store discretized backbone weights')
    parser.add_argument('--precision-head', type=int, default=16, help='Bits to store discretized head weights')
    parser.add_argument('--img-size', type=int, default=640, help='Train & validation image size in pixels')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--batch-size-test', type=int, default=8, help='Test batch size')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    return args

def main(args):
    # Select GPU or CPU
    if torch.cuda.is_available():
        USE_CUDA = True
        device = torch.device('cuda')
        LOGGER.info('CUDA on')
    else:
        USE_CUDA = False
        device = torch.device('cpu')
        LOGGER.info('CUDA off')

    # Import config_dict
    config_dict = import_config_dict()

    # Train
    train(args=args, config_dict=config_dict, device=device, USE_CUDA=USE_CUDA)

if __name__ == '__main__':
    args = parse_args()
    main(args)