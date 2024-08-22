"""
Build the network model using common YOLOv5 modules
"""

import sys
import logging
import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # analog_ai_object_detection directory

from models.yolov5.common import *
from utils.torch_utils import time_sync, fuse_conv_and_bn, scale_img

# Add ROOT to PATH
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# Logger
LOGGER = logging.getLogger("MainLogger")

class Detect(nn.Module):
    stride = None           # strides computed during build
    onnx_dynamic = False    # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.num_classes = nc
        self.outputs_per_anchor = nc + 5            # Number of outputs per anchor
        self.num_detect_layers = len(anchors)       # Number of detection layers
        self.num_anchors = len(anchors[0]) // 2     # Number of anchors
        self.grid = [torch.zeros(1)] * self.num_detect_layers
        a = torch.tensor(anchors).float().view(self.num_detect_layers, -1, 2)
        self.register_buffer('anchors', a)                                                              # shape(num_detect_layers, num_anchors, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.num_detect_layers, 1, -1, 1, 1, 2))     # shape(num_detect_layers, 1, num_anchors, 1, 1, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.outputs_per_anchor * self.num_anchors, 1) for x in ch) # Output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.num_detect_layers):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_anchors, self.outputs_per_anchor, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    self.grid[i] = self.grid[i].to(y.device)
                    self.stride[i] = self.stride[i].to(y.device)
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.num_anchors, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.outputs_per_anchor))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg, ch=3, nc=None, anchors=None):   # model, input channels, number of classes
        super().__init__()
        self.model_config = cfg

        # Define model
        ch = self.model_config['ch'] = self.model_config.get('ch', ch)        # input channels (=3 for RGB)

        # Override the default number of classes in cfg with the number of classes provided by the dataset
        if nc and nc != self.model_config['nc']:
            LOGGER.info(f"Overriding cfg nc={self.model_config['nc']} with nc={nc}")
            self.model_config['nc'] = nc

        if anchors and isinstance(anchors, (int, float)):
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)

        self.model, self.save = parse_model(d=deepcopy(self.model_config), ch=[ch])
        self.names = [str(i) for i in range(self.model_config['nc'])]  # default names
        self.inplace = self.model_config.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Last module of the model -> Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # Forward
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        self.info()

    def forward(self, x, augment=False, profile=False, visualize=False, calc_MAC=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize, calc_MAC)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None

    def _forward_once(self, x, profile=False, visualize=False, calc_MAC=False):
        y, dt = [], []  # outputs
        self.cumulative_num_MAC = 0 # Cumulative number of MAC operations
        self.i_conv2d = 0           # Number of 2D Convolutional Layers (nn.Conv2d)
        self.i_MaxPool2d = 0        # Number of 2D Max Pool Layers (nn.MaxPool2d)
        if calc_MAC:
            LOGGER.info(f"Calculating the number of MAC operations...")
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if type(m) not in [nn.Upsample, Concat, Detect]:
                x = x.type(next(m.parameters()).dtype)
            x = m(x)  # run

            if isinstance(x, list): # Input of Detect is a list
                idx_input = 0   # Index of the input list

            if calc_MAC:    # Calculate number of MAC operations
                for layer in m.modules():
                    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                        if isinstance(layer, nn.Conv2d):        # If Conv2d
                            self.i_conv2d += 1
                            kernel_h, kernel_w = layer.kernel_size[0], layer.kernel_size[1] # Kernel sizes (h, w)
                            ch_in, ch_out = layer.in_channels, layer.out_channels           # Input and output channels

                        elif isinstance(layer, nn.MaxPool2d):   # If MaxPool2d
                            self.i_MaxPool2d += 1
                            kernel_h = kernel_w = layer.kernel_size # Kernel sizes (h, w)
                            ch_in = ch_out = 1  # Neutral element, as MaxPool2d does not have trainable weights

                        if isinstance(m, Detect):   # If Detect
                            h_out, w_out = x[idx_input].size(3), x[idx_input].size(2) # h and w of the layer output (img)
                            idx_input += 1
                        else:
                            h_out, w_out = x.size(-1), x.size(-2)   # h and w of the layer output (img)
                        num_MAC = kernel_h * kernel_w * ch_in * ch_out * h_out * w_out # Number of MAC in this layer
                        self.cumulative_num_MAC += num_MAC  # Cumulative number of MAC operations

                        ''' For printing/debugging purposes
                        LOGGER.info(f"Module name: {m.__class__.__name__}")
                        LOGGER.info(f"Layer name: {layer.__class__.__name__}")
                        LOGGER.info(f"Total number of Conv2d layers: {self.i_conv2d}")
                        LOGGER.info(f"Total number of MaxPool2d layers: {self.i_MaxPool2d}")
                        LOGGER.info(f"Number of MAC operations: {num_MAC}")
                        LOGGER.info(f"Total number of MAC operations: {self.cumulative_num_MAC}")
                        LOGGER.info(f"Output dimensions: {x[idx_input-1].size()}" if isinstance(x, list) else f"{x.size()}")
                        LOGGER.info(f"Layer specifications: {layer}")
                        LOGGER.info(f"--------------------------------------------------------------------------------")
                        '''

            y.append(x if m.i in self.save else None)  # save output
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.block_type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.num_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.num_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.num_anchors, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=480):
        # Print model information with type(img_size) = int or list, i.e. img_size = 640 or img_size = [640, 320]
        num_parameters = sum(x.numel() for x in self.parameters())
        num_gradients = sum(x.numel() for x in self.parameters() if x.requires_grad)
        x_dummy = torch.zeros((1, 3, img_size, img_size)) # Dummy image tensor to calculate cum_num_MAC
        self(x=x_dummy, calc_MAC=True)
        cum_num_MAC = self.cumulative_num_MAC

        try:  # FLOPs
            from thop import profile
            stride = max(int(self.stride.max()), 32) if hasattr(self, 'stride') else 32
            img = torch.zeros((1, self.model_config.get('ch', 3), stride, stride),
                              device=next(self.parameters()).device)  # input
            flops = profile(self, inputs=(img,), verbose=False)[0] / 1E9 * 2                    # stride GFLOPs
            img_size = img_size if isinstance(img_size, list) else [img_size, img_size]         # expand if int/float
            fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)        # 480x480 GFLOPs
        except (ImportError, Exception):
            fs = ''

        LOGGER.info(f"Model Summary: {len(list(self.modules()))} layers, {num_parameters} parameters, "
                    f"{num_gradients} gradients{fs}, {round(cum_num_MAC / 10**9, 4)}\u202210⁹ MAC operations"
                    f"with an image size of {img_size}")

def parse_model(d, ch):   # model dict, input channels
    # Logger prepare the x-axis of the listing
    LOGGER.info('%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))

    anchors, num_classes = d['anchors'], d['nc']                                    # anchors, number of classes
    multiplier_d, multiplier_w = d['depth_multiple'], d['width_multiple']           # depth multiplier, width multiplier
    num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    num_outputs = num_anchors * (num_classes + 5)
    # number of outputs = number of anchors * (number of classes + 5 predictions (x_center, y_center, h, w, IoU))

    layers = []         # list of layers
    save = []           # savelist
    ch_out = ch[-1]     # ch_out

    # Iteration through the layers of the network model
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):     # from, number of modules, module, args
        m = eval(m) if isinstance(m, str) else m    # Eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a      # Eval strings in args
            except:
                pass
        n = n_ = max(round(n * multiplier_d), 1) if n > 1 else n    # Depth decrease for yolov5s,yolov5m/ increase for yolov5x

        if m in [Focus, Conv, C3, SPP]:
            ch_in = ch[f]               # input channel
            ch_out = args[0]            # output channel
            if ch_out != num_outputs:   # if not network output
                ch_out = math.ceil((ch_out * multiplier_w) / 8) * 8     # Output dimension must be divisible by 8

            args = [ch_in, ch_out, *args[1:]]
            if m is C3:
                args.insert(2, n)   # number of repeats
                n = 1               # Reset n

        elif m is Concat:   # Concatenation layer
            ch_out = sum([ch[x] for x in f])

        elif m is Detect:   # Detection layer (last layer)
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)

        else:
            ch_out = ch[f]

        module = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')           # Module type
        np = sum([x.numel() for x in module.parameters()])  # Number of parameters

        # Assign values to 'module'
        module.i = i            # Index
        module.f = f            # 'from' index
        module.block_type = t   # Module layer blocks
        module.num_params = np  # number of parameters

        # Print the values of the layer
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))

        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)   # Append index of concat- and detect layers to savelist
        layers.append(module)   # Append to layers list
        if i == 0:
            ch = []
        ch.append(ch_out)

    return nn.Sequential(*layers), sorted(save)