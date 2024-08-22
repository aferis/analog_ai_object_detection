"""
General utils
"""

import os
import torch
import math
import time
import logging
import platform
import torchvision
import numpy as np
from pathlib import Path
import argparse

from utils.metrics import box_iou

# Logger
LOGGER = logging.getLogger("MainLogger")

def quantize_normalized(weight, device, num_bits=8, partition='exponential', rounding='mean'):
    max = torch.abs(weight).max()
    n = 2 ** (num_bits - 1)
    signs = torch.sign(weight)
    x = torch.abs(torch.flatten(weight)) / max  # Normalized values between 0 and 1
    if partition == 'linear':
        x0 = 1 / n
        q = (1 - x0) / (n - 1)
        xx = [0] + [x0 + q*i for i in range(n)]
    elif partition == 'exponential':
        x0 = (1 / n) ** 1.5
        q = x0 ** (-1 / (n - 1))
        xx = [0] + [x0 * (q ** i) for i in range(n)]

    weight_q = torch.zeros(weight.size()).to(device)
    for i in range(len(xx) - 1):
        ii = torch.logical_and(xx[i] < x, x <= xx[i+1])
        ii = torch.reshape(ii, weight_q.shape)
        if rounding == 'floor':
            weight_q[ii] = xx[i]
        elif rounding == 'ceil':
            weight_q[ii] = xx[i+1]
        elif rounding == 'mean':
            xx_mean = (xx[i] + xx[i+1]) / 2
            weight_q[ii] = xx_mean

    return torch.reshape(weight_q, weight.shape) * signs * max

def quantize_standard(weight, device, min=None, max=None, num_bits=8, partition='linear', rounding='nearest'):
    if min is None and max is None:
        min = weight.min().item()
        max = weight.max().item()
        x = torch.flatten(weight)
        if min == max:
            return weight
    elif min and max:
        divident = abs(max) if abs(max) >= abs(min) else abs(min)
        if num_bits > 8:
            return weight if weight.min().item() >= min and weight.max().item() <= max else weight / (abs(weight).max() / divident)
        x = torch.flatten(weight) / (abs(weight).max() / divident)
    n = 2 ** num_bits
    if partition == 'linear':
        q = (max - min) / (n - 1)
        xx = [min + q*i for i in range(n)]
    elif partition == 'exponential':
        pass
    weight_q = torch.zeros(weight.size()).to(device)
    for i in range(len(xx) - 1):
        ii_flattened = torch.logical_and(xx[i] < x, x <= xx[i + 1])
        ii = torch.reshape(ii_flattened, weight_q.shape)
        if rounding == 'floor':
            weight_q[ii] = xx[i]
        elif rounding == 'ceil':
            weight_q[ii] = xx[i + 1]
        elif rounding == 'nearest':
            ii_lower = torch.logical_and(ii_flattened, torch.abs(x - xx[i]) < torch.abs(x - xx[i + 1]))
            ii_lower = torch.reshape(ii_lower, weight_q.shape)
            ii_upper = torch.logical_and(ii_flattened, torch.abs(x - xx[i]) > torch.abs(x - xx[i + 1]))
            ii_upper = torch.reshape(ii_upper, weight_q.shape)
            weight_q[ii_lower] = xx[i]
            weight_q[ii_upper] = xx[i + 1]

    return torch.reshape(weight_q, weight.shape)

def str2dtype(v):
    if isinstance(v, torch.dtype):
        return v
    if v.lower() in ('float64', 'torch.float64', 'torch.double', 'f64', 'fp64', 'double', '64-bit', '64'):
        return torch.float64
    if v.lower() in ('float32', 'torch.float32', 'torch.float', 'f32', 'fp32', 'float', '32-bit', '32'):
        return torch.float32
    if v.lower() in ('float16', 'torch.float16', 'torch.half', 'f16', 'fp16', 'half', '16-bit', '16', 'binary16') and torch.cuda.is_available():
        return torch.half
    if v.lower() in ('float16', 'torch.float16', 'torch.half', 'f16', 'fp16', 'half', '16-bit', '16', 'binary16') and not torch.cuda.is_available():
        raise TypeError('Half precision is only supported in CUDA.')
    if v.lower() in ('bfloat16', 'torch.bfloat16', 'bf16', 'brain floating point') and torch.cuda.is_available():
        return torch.bfloat16
    if v.lower() in ('bfloat16', 'torch.bfloat16', 'bf16', 'brain floating point') and not torch.cuda.is_available():
        raise TypeError('Half precision is only supported in CUDA.')
    else:
        raise TypeError('torch.dtype value expected.')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def user_config_dir(dir='Ultralytics', env_var='YOLOV5_CONFIG_DIR'):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # Use environment variable
    else:
        cfg = {'Windows': 'AppData/Roaming', 'Linux': '.config', 'Darwin': 'Library/Application Support'}  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), '')         # OS-specific config dir
        path = (path if is_writeable(path) else Path('/tmp')) / dir # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # Make if required
    return path

def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / 'tmp.txt'
        try:
            with open(file, 'w'):  # Open file with write permissions
                pass
            file.unlink()  # Remove file
            return True
        except IOError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # Possible issues on Windows

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)              # Labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)           # Labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)    # Occurrences per class

    weights[weights == 0] = 1                       # Replace empty bins with 1
    weights = 1 / weights                           # Number of targets per class
    weights /= weights.sum()                        # Normalize
    return torch.from_numpy(weights)

def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # Calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5            # Number of classes
    xc = prediction[..., 4] > conf_thres    # Candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096    # (Pixels) minimum and maximum box width and height
    max_nms = 30000             # Maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0           # Seconds to quit after
    redundant = True            # Redundant detections
    multi_label &= nc > 1       # Multiple labels per box (adds 0.5ms/img)
    merge = False               # Use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # Confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute confidence
        x[:, 5:] *= x[:, 4:5]   # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            # x of size(num_filtered_bboxes * nc (each bbox contains confidence scores for each class!), (x_center, y_center, w, h, conf, class)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # Number of boxes with unique [..., conf, class]-combination!
        if not n:  # No boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)         # Classes
        boxes, scores = x[:, :4] + c, x[:, 4]               # Boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)   # NMS -> returns tensor i with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores (considering IoUs)
        if i.shape[0] > max_det:    # Limit detections
            i = i[:max_det]     # Take only the best first <max_det> bbox indices
        if merge and (1 < n < 3E3): # Merge NMS (boxes merged using weighted mean)
            # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # Box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # Merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # Require redundancy

        output[xi] = x[i]   # size(num_filtered_bboxes, (x_center, y_center, w, h, conf, class)
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # Time limit exceeded

    return output

def strip_optimizer(f='best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")

################################# Box conversions ######################################################################

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # Top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # Top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # Bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # Bottom right y
    return y

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # Top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # Top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # Bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # Bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y

def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy

def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh