"""
Validate/test a trained model accuracy on a custom dataset

Usage (two options):
    -> called by train.py for validation & calculation of mAPs during training
    -> should be run directly for inference with pre-trained weights
    e.g. $ python val.py --weights path/to/best.pt --img-size 480 --batch-size 8 --conf-thres 0.25 --iou-thres 0.45
"""

# Default libraries & packages
import os
import sys
import yaml
import time
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from threading import Thread

# Inside analog_ai_object_detection
from global_settings import import_config_dict
from utils.loss import ComputeLoss
from utils.datasets import create_dataloader
from utils.plots import plot_images, output_to_target
from utils.metrics import ConfusionMatrix, box_iou, ap_per_class
from utils.torch_utils import time_sync
from utils.general import non_max_suppression, scale_coords, xywh2xyxy, str2bool
from models.yolov5.yolo import Model
from models.yolov5.experimental import attempt_load
from rpu_settings import rpu_config, tune_rpu_hyperpamaters

# AIHWKIT Imports
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn.modules.conv import *
from aihwkit.simulator.configs.configs import *
from aihwkit.simulator.configs.devices import *

# Logger
LOGGER = logging.getLogger("MainLogger")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # analog_ai_object_detection directory

# Add ROOT to PATH
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

timestr = time.strftime("%Y.%m.%d-%H:%M:%S")

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

@torch.no_grad()
def run(data,
        seed=None,
        seed_count=1,
        weights=None,       # model.pt path(s)
        batch_size=16,      # Batch size
        imgsz=640,          # Inference size (pixels)
        conf_thres=0.001,   # Confidence threshold
        iou_thres=0.6,      # NMS IoU threshold
        device='',          # Cuda device, i.e. 0 or 0,1,2,3 or cpu
        augment=False,      # Augmented inference
        verbose=False,      # Verbose output
        save_txt=False,     # Save results to *.txt
        save_hybrid=False,  # Save label+prediction hybrid results to *.txt
        model=None,
        analog=False,
        dataloader=None,
        save_dir='',
        plots=True,
        compute_loss=None,
        config_dict=None,
        USE_CUDA=False,
        cluster=False):

    # Check if training or testing
    training = model is not None

    # Get number of classes, and class names
    num_classes, class_names = data['nc'], data['names']
    class_names = {k: v for k, v in enumerate(class_names)}
    assert len(class_names) == num_classes, f'{len(class_names)} class names were found for nc={num_classes}'

    # Get YOLO network architecture
    architecture = config_dict['NETWORK']['architecture']
    model_config_file = os.path.join(ROOT, 'models/yolov5/', architecture) + '.yaml'
    if isinstance(model_config_file, (str, Path)):
        with open(model_config_file, mode="r") as f:
            model_config_dict = yaml.safe_load(f)

    # Length of backbone and head
    num_bb_modules = len(model_config_dict['backbone'])  # Number of backbone modules
    num_head_modules = len(model_config_dict['head'])  # Number of head modules

    if not training:
        # Test set directory
        if cluster:
            test_path = data['test_cluster']
        else:
            test_path = data['test_local']

        # Load model
        checkpoint = torch.load(weights)
        epoch = checkpoint['epoch']
        best_fitness = checkpoint['best_fitness']
        model = checkpoint['model']
        model.to(device).float()
        compute_loss = ComputeLoss(model)  # Init loss class

    if not dataloader:
        # TestLoader
        dataloader, test_dataset = create_dataloader(path=test_path, imgsz=imgsz, batch_size=batch_size,
                                                     stride=int(model.stride.max()), hyp=config_dict['HYPERPARAMETER'],
                                                     pad=0.5, prefix='Test: ')

    if analog and not isinstance(model.model, AnalogSequential):
        # Setup RPU configuration
        RPU_config = config_dict['RPU']['configuration']
        RPU_params = config_dict['RPU']['value']
        device_config = rpu_config(RPU_config, RPU_params)

        # Modifying RPU parameters
        tune_rpu_param = config_dict['RPU']['tune_rpu_param']
        if tune_rpu_param and isinstance(device_config, InferenceRPUConfig):
            tune_rpu_hyperpamaters(device_config, config_dict['RPU'], seed)
        if tune_rpu_param and isinstance(device_config.device, IdealDevice):  # Specify only seed for IdealDevice
            device_config.device.construction_seed = seed
        elif tune_rpu_param and 'preset' in RPU_config.lower():  # Specify only seed & is_perfect for presets
            device_config.device.construction_seed = seed
            device_config.forward.is_perfect = config_dict['RPU']['fw_is_perfect']
            device_config.backward.is_perfect = config_dict['RPU']['bw_is_perfect']
        elif tune_rpu_param and not isinstance(device_config, FloatingPointRPUConfig):
            tune_rpu_hyperpamaters(device_config, config_dict['RPU'], seed)

        model = convert_to_analog(model, device_config, weight_scaling_omega=0.6)
        model.to(device)

    # Configure
    model.eval()
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    v_iou = torch.linspace(0.5, 0.95, 10).to(device)  # IoU vector for mAP@0.5:0.95
    n_iou = v_iou.numel()   # Total number of elements in IoU vector

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=num_classes)
    class_map = list(range(1000))
    s = ('%50s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt = [0.0, 0.0, 0.0]    # Delta t [t_prep, t_inference, t_non_max_suppression]
    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        img = img.to(device, non_blocking=True).float() # uint8 to float32
        img = img / 255.0  # 0-255 to 0.0-1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1    # Data preparation time

        # Inference
        ## out: shape(num_val_img, number of all proposed/possible b_boxes, num_classes + 5)
        ## train_out{list: num_output_layers}: shape(num_val_img, ch, downsampled_h, downsampled_w, num_classes + 5)
        t3 = time_sync()
        out, train_out = model(img.type_as(next(model.parameters())), augment=augment)    # Inference and loss outputs
        t4 = time_sync()
        dt[1] += t4 - t3

        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
        # Apply Non-Maximum Suppression (NMS)
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)           # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []    # for autolabelling
        t5 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True) # Returns filtered b_box proposals for all images
        # Each element of the list 'out' has shape(num_filtered_bboxes, (x_center, y_center, w, h, conf, class)
        t6 = time_sync()
        dt[2] += t6 - t5

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, n_iou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1]) # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)              # native-space labels
                correct = process_batch(predn, labelsn, v_iou)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], n_iou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

        # Plot example validation batch images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, class_names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, class_names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        if seed_count > 1:
            p, r, ap, f1, ap_class = ap_per_class(*stats, seed=seed, plot=plots, save_dir=save_dir, names=class_names)
        else:
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=class_names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=num_classes)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (num_classes < 50 and not training)) and num_classes > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (class_names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(round(x,4) for x in dt)                        # Total speeds [s]
    t_per_img = tuple(round(x / seen * 1E3, 4) for x in dt)   # Speeds [ms] per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Total speed: %.1fs pre-process, %.1fs inference, %.1fs NMS with conf-thres:{conf_thres} '
                    f'and IoU thres:{iou_thres} for {seen} images' % t)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t_per_img)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(class_names.values()))

    # Return results
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}")
    maps = np.zeros(num_classes) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, (t + t_per_img)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=str2bool, default=False, help='Whether to use GPU cluster')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path')
    parser.add_argument('--analog', type=str2bool, default=False, help='Whether to apply analog inference')
    parser.add_argument('--img-size', type=int, default=640, help='Test image size in pixels')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--augment', type=str2bool, default=False, help='Augmented inference')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Report mAP by class')
    args = parser.parse_args()
    return args

def main(args):
    # Select GPU or CPU
    if torch.cuda.is_available():
        USE_CUDA = True
        device = torch.device('cuda')
    else:
        USE_CUDA = False
        device = torch.device('cpu')

    # Import config_dict
    config_dict = import_config_dict()

    # Create test directory (unique to the training that contains best.pt)
    RPU_config = config_dict['RPU']['configuration']
    is_perfect = config_dict['RPU']['fw_is_perfect']
    weight_noise_type = config_dict['RPU']['fw_w_noise_type'].split('.', 1)[1]
    inp_res = config_dict['RPU']['fw_inp_res_bits']
    out_res = config_dict['RPU']['fw_out_res_bits']
    inp_noise = config_dict['RPU']['fw_inp_noise']
    w_noise = config_dict['RPU']['fw_w_noise']
    out_noise = config_dict['RPU']['fw_out_noise']
    g_max = config_dict['RPU']['g_max']
    save_dir = os.path.dirname(os.path.dirname(args.weights))
    if args.analog and any(x in save_dir.lower() for x in ['d_16bb', '16head', 'standard_']):
        if 'preset' in RPU_config.lower():
            save_dir = Path(os.path.join(Path(save_dir), f'test_analog/{RPU_config}/{timestr[:-3]}_'
                                                         f'conf{args.conf_thres}_iou{args.iou_thres}'))
        else:
            save_dir = Path(os.path.join(Path(save_dir), f'test_analog/{RPU_config}/{timestr[5:-3]}_{args.conf_thres}_'
                                                         f'{args.iou_thres}_{is_perfect}_{weight_noise_type}_'
                                                         f'{inp_res}bit_{out_res}bit_iN{inp_noise}_wN{w_noise}_'
                                                         f'oN{out_noise}_gmax{g_max}'))
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save input arguments & RPU configs
        with open(save_dir / 'args.yaml', 'w') as f:
            yaml.safe_dump(vars(args), f, sort_keys=False)
        with open(save_dir / 'rpu.yaml', 'w') as f:
            yaml.safe_dump(config_dict['RPU'], f, sort_keys=False)
    else:
        save_dir = Path(os.path.join(Path(save_dir), f'test/{timestr}_conf{args.conf_thres}_iou{args.iou_thres}'))
        save_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(module)s:%(lineno)d] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)

    # Create a log file
    log_file = save_dir / 'log.txt'
    fh = logging.FileHandler(filename=log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)

    # Load the dataset configuration file as a dict
    dataset_config_file = os.path.join(ROOT, config_dict['DATASET']['config_dataset'])
    if isinstance(dataset_config_file, (str, Path)):
        with open(dataset_config_file, mode="r") as f:
            dataset_config_dict = yaml.safe_load(f)

    # Check if the necessary elements exist in the dataset config file
    assert 'test' or 'test_local' or 'test_cluster' in dataset_config_dict, "'test' key is missing"
    assert 'nc' in dataset_config_dict, "'nc' key (number of classes) is missing."
    if 'names' not in dataset_config_dict:
        dataset_config_dict['names'] = [f'class{i}' for i in range(dataset_config_dict['nc'])]

    results, _, t = run(seed=1, data=dataset_config_dict, weights=args.weights, batch_size=args.batch_size,
                        imgsz=args.img_size, device=device, conf_thres=args.conf_thres, iou_thres=args.iou_thres,
                        analog=args.analog, save_dir=save_dir, config_dict=config_dict, USE_CUDA=USE_CUDA,
                        cluster=args.cluster)

    loss_keys = ['test/precision', 'test/recall', 'test/mAP_0.5', 'test/mAP_0.5:0.95',
                 'test/box_loss', 'test/obj_loss', 'test/cls_loss',
                 't_total[s]/pre-process', 't_total[s]/inference', 't_total[s]/NMS',
                 't_per_img[ms]/pre-process', 't_per_img[ms]/inference', 't_per_img[ms]/NMS']
    loss_dict = {k: v for k, v in zip(loss_keys, list(results) + list(t))}
    with open(save_dir / 'test.csv', 'w') as f:
        f.write((('%20s,' * len(loss_dict) % tuple(loss_keys)).rstrip(',') + '\n') +
                ('%20.5g,' * len(loss_dict) % tuple(list(results) + list(t))).rstrip(','))

if __name__ == "__main__":
    args = parse_args()
    main(args)