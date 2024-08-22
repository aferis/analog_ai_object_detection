"""
Assign hyperparameters to desired data types
"""

import configparser
import json

def create_config_dict(config):
    dict = {}
    for section in config.sections():
        dict[section] = {}
        for key, val in config.items(section):
            dict[section][key] = val
    return dict

def import_config_dict():
    config = configparser.ConfigParser()
    config.read('config/config_dataset.ini')
    config_dict = create_config_dict(config)

    config.read('config/config_image.ini')
    config_dict = create_config_dict(config)

    config.read('config/config_hyperparameter.ini')
    config_dict.update(create_config_dict(config))

    config.read('config/config_net.ini')
    config_dict.update(create_config_dict(config))

    config.read('config/config_savepath.ini')
    config_dict.update(create_config_dict(config))

    config.read('config/config_rpu.ini')
    config_dict.update(create_config_dict(config))

    config.read('config/config_optimizer.ini')
    config_dict.update(create_config_dict(config))

    config.read('config/config_stopper.ini')
    config_dict.update(create_config_dict(config))

    config.read('config/config_scheduler.ini')
    config_dict.update(create_config_dict(config))

    # Image
    config_dict['IMAGE']['augmentation'] = str2bool(config_dict['IMAGE']['augmentation'])
    config_dict['IMAGE']['dataload_workers'] = int(config_dict['IMAGE']['dataload_workers'])

    # Hyperparameters
    config_dict['HYPERPARAMETER']['learning_rate'] = float(config_dict['HYPERPARAMETER']['learning_rate'])
    config_dict['HYPERPARAMETER']['momentum'] = float(config_dict['HYPERPARAMETER']['momentum'])
    config_dict['HYPERPARAMETER']['weight_decay'] = float(config_dict['HYPERPARAMETER']['weight_decay'])
    config_dict['HYPERPARAMETER']['use_seed'] = str2bool(config_dict['HYPERPARAMETER']['use_seed'])

    ## Thresholds
    config_dict['HYPERPARAMETER']['anchor_t'] = float(config_dict['HYPERPARAMETER']['anchor_t'])
    config_dict['HYPERPARAMETER']['conf_t'] = float(config_dict['HYPERPARAMETER']['conf_t'])
    config_dict['HYPERPARAMETER']['iou_t'] = float(config_dict['HYPERPARAMETER']['iou_t'])

    ## Scaling parameters
    config_dict['HYPERPARAMETER']['box_loss_gain'] = float(config_dict['HYPERPARAMETER']['box_loss_gain'])
    config_dict['HYPERPARAMETER']['cls_loss_gain'] = float(config_dict['HYPERPARAMETER']['cls_loss_gain'])
    config_dict['HYPERPARAMETER']['obj_loss_gain'] = float(config_dict['HYPERPARAMETER']['obj_loss_gain'])
    config_dict['HYPERPARAMETER']['label_smoothing'] = float(config_dict['HYPERPARAMETER']['label_smoothing'])

    ## Warmup parameters
    config_dict['HYPERPARAMETER']['warmup_epochs'] = float(config_dict['HYPERPARAMETER']['warmup_epochs'])
    config_dict['HYPERPARAMETER']['warmup_bias_lr'] = float(config_dict['HYPERPARAMETER']['warmup_bias_lr'])
    config_dict['HYPERPARAMETER']['warmup_momentum'] = float(config_dict['HYPERPARAMETER']['warmup_momentum'])

    ## BCELoss positive_weight parameters
    config_dict['HYPERPARAMETER']['cls_pw'] = float(config_dict['HYPERPARAMETER']['cls_pw'])
    config_dict['HYPERPARAMETER']['obj_pw'] = float(config_dict['HYPERPARAMETER']['obj_pw'])

    ## Focal loss gamma
    config_dict['HYPERPARAMETER']['fl_gamma'] = float(config_dict['HYPERPARAMETER']['fl_gamma'])

    ## Image mosaic and mixup (probabilities)
    config_dict['HYPERPARAMETER']['mosaic'] = float(config_dict['HYPERPARAMETER']['mosaic'])
    config_dict['HYPERPARAMETER']['mixup'] = float(config_dict['HYPERPARAMETER']['mixup'])

    ## Segment copy-paste (probability)
    config_dict['HYPERPARAMETER']['copy_paste'] = float(config_dict['HYPERPARAMETER']['copy_paste'])

    ## Parameters for image augmentations
    config_dict['HYPERPARAMETER']['degrees'] = float(config_dict['HYPERPARAMETER']['degrees'])
    config_dict['HYPERPARAMETER']['translate'] = float(config_dict['HYPERPARAMETER']['translate'])
    config_dict['HYPERPARAMETER']['scale'] = float(config_dict['HYPERPARAMETER']['scale'])
    config_dict['HYPERPARAMETER']['shear'] = float(config_dict['HYPERPARAMETER']['shear'])
    config_dict['HYPERPARAMETER']['perspective'] = float(config_dict['HYPERPARAMETER']['perspective'])
    config_dict['HYPERPARAMETER']['hsv_h'] = float(config_dict['HYPERPARAMETER']['hsv_h'])
    config_dict['HYPERPARAMETER']['hsv_s'] = float(config_dict['HYPERPARAMETER']['hsv_s'])
    config_dict['HYPERPARAMETER']['hsv_v'] = float(config_dict['HYPERPARAMETER']['hsv_v'])
    config_dict['HYPERPARAMETER']['flipud'] = float(config_dict['HYPERPARAMETER']['flipud'])
    config_dict['HYPERPARAMETER']['fliplr'] = float(config_dict['HYPERPARAMETER']['fliplr'])

    # RPU configuration
    config_dict['RPU']['parameters'] = list(config_dict['RPU']['parameters'].split(","))
    config_dict['RPU']['value'] = json.loads(config_dict['RPU']['value'])

    # RPU Hyperparameters - Main boolean
    config_dict['RPU']['tune_rpu_param'] = str2bool(config_dict['RPU']['tune_rpu_param'])
    config_dict['RPU']['fw_bw_identical'] = str2bool(config_dict['RPU']['fw_bw_identical'])

    # RPU Hyperparameters - Forward pass
    config_dict['RPU']['fw_inp_bound'] = float(config_dict['RPU']['fw_inp_bound'])
    config_dict['RPU']['fw_inp_noise'] = float(config_dict['RPU']['fw_inp_noise'])
    config_dict['RPU']['fw_inp_res_bits'] = int(config_dict['RPU']['fw_inp_res_bits'])
    config_dict['RPU']['fw_is_perfect'] = str2bool(config_dict['RPU']['fw_is_perfect'])
    config_dict['RPU']['fw_out_bound'] = float(config_dict['RPU']['fw_out_bound'])
    config_dict['RPU']['fw_out_noise'] = float(config_dict['RPU']['fw_out_noise'])
    config_dict['RPU']['fw_out_res_bits'] = int(config_dict['RPU']['fw_out_res_bits'])
    config_dict['RPU']['fw_w_noise'] = float(config_dict['RPU']['fw_w_noise'])

    # Additional Hyperparameters - Inference
    config_dict['RPU']['g_max'] = float(config_dict['RPU']['g_max'])

    # RPU Hyperparameters - Backward Propagation
    config_dict['RPU']['bw_inp_bound'] = float(config_dict['RPU']['bw_inp_bound'])
    config_dict['RPU']['bw_inp_noise'] = float(config_dict['RPU']['bw_inp_noise'])
    config_dict['RPU']['bw_inp_res_bits'] = int(config_dict['RPU']['bw_inp_res_bits'])
    config_dict['RPU']['bw_is_perfect'] = str2bool(config_dict['RPU']['bw_is_perfect'])
    config_dict['RPU']['bw_out_bound'] = float(config_dict['RPU']['bw_out_bound'])
    config_dict['RPU']['bw_out_noise'] = float(config_dict['RPU']['bw_out_noise'])
    config_dict['RPU']['bw_out_res_bits'] = int(config_dict['RPU']['bw_out_res_bits'])
    config_dict['RPU']['bw_w_noise'] = float(config_dict['RPU']['bw_w_noise'])

    # RPU Hyperparameters - Device
    config_dict['RPU']['device_dw_min'] = float(config_dict['RPU']['device_dw_min'])
    config_dict['RPU']['device_up_down'] = float(config_dict['RPU']['device_up_down'])
    config_dict['RPU']['device_dw_min_dtod'] = float(config_dict['RPU']['device_dw_min_dtod'])
    config_dict['RPU']['device_dw_min_std'] = float(config_dict['RPU']['device_dw_min_std'])
    config_dict['RPU']['device_up_down_dtod'] = float(config_dict['RPU']['device_up_down_dtod'])
    config_dict['RPU']['device_w_max_dtod'] = float(config_dict['RPU']['device_w_max_dtod'])
    config_dict['RPU']['device_w_min_dtod'] = float(config_dict['RPU']['device_w_min_dtod'])

    # Scheduler
    config_dict['SCHEDULER']['use_scheduler'] = str2bool(config_dict['SCHEDULER']['use_scheduler'])
    config_dict['SCHEDULER']['lrf'] = float(config_dict['SCHEDULER']['lrf'])
    config_dict['SCHEDULER']['milestones'] = json.loads(config_dict['SCHEDULER']['milestones'])
    config_dict['SCHEDULER']['gamma'] = float(config_dict['SCHEDULER']['gamma'])

    # Stopper
    config_dict['STOPPER']['use_stopper'] = str2bool(config_dict['STOPPER']['use_stopper'])
    config_dict['STOPPER']['patience'] = int(config_dict['STOPPER']['patience'])

    return config_dict

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
