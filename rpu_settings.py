"""
It is used to adjust the configuration of the different devices.
Then the analog neural network can be trained in these devices.
"""

from aihwkit.simulator.configs import SingleRPUConfig, \
    FloatingPointRPUConfig, InferenceRPUConfig, UnitCellRPUConfig, DigitalRankUpdateRPUConfig
from aihwkit.simulator.configs.devices import IdealDevice, ConstantStepDevice, \
    FloatingPointDevice, TransferCompound, SoftBoundsDevice, ExpStepDevice, \
    LinearStepDevice, PowStepDevice, SoftBoundsPmaxDevice, MixedPrecisionCompound
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType, \
    WeightNoiseType, WeightClipType, WeightModifierType, PulseType
from aihwkit.simulator.presets import TikiTakaCapacitorPreset, \
    TikiTakaEcRamPreset, TikiTakaIdealizedPreset, TikiTakaReRamESPreset, \
    TikiTakaReRamSBPreset, TikiTakaEcRamMOPreset, MixedPrecisionReRamESPreset, \
    MixedPrecisionReRamSBPreset, MixedPrecisionCapacitorPreset, MixedPrecisionEcRamMOPreset, \
    MixedPrecisionGokmenVlasovPreset, MixedPrecisionPCMPreset, MixedPrecisionIdealizedPreset, \
    MixedPrecisionEcRamPreset
from aihwkit.simulator.noise_models import PCMLikeNoiseModel, \
    GlobalDriftCompensation
from global_settings import import_config_dict

def tune_rpu_hyperpamaters(rpu, config_rpu, seed):
    # Parameters - Forward pass
    rpu.forward.bound_management = eval(config_rpu['fw_bound_management'])
    rpu.forward.inp_bound = config_rpu['fw_inp_bound']
    rpu.forward.inp_noise = config_rpu['fw_inp_noise']
    rpu.forward.inp_res = 1 / (2**config_rpu['fw_inp_res_bits'] - 2)
    rpu.forward.is_perfect = config_rpu['fw_is_perfect']
    rpu.forward.noise_management = eval(config_rpu['fw_noise_management'])
    rpu.forward.out_bound = config_rpu['fw_out_bound']
    rpu.forward.out_noise = config_rpu['fw_out_noise']
    rpu.forward.out_res = 1 / (2**config_rpu['fw_out_res_bits'] - 2)
    rpu.forward.w_noise = config_rpu['fw_w_noise']
    rpu.forward.w_noise_type = eval(config_rpu['fw_w_noise_type'])
    if isinstance(rpu, InferenceRPUConfig):
        # Specify the noise model to be used for inference only
        # g_max: the value the absolute max of the weights will be mapped to.
        rpu.noise_model = PCMLikeNoiseModel(g_max=config_rpu['g_max'])
        return

    # Parameters - Backward propagation
    if config_rpu['fw_bw_identical']:
        rpu.backward = rpu.forward
    else:
        rpu.backward.inp_bound = config_rpu['bw_inp_bound']
        rpu.backward.inp_noise = config_rpu['bw_inp_noise']
        rpu.backward.inp_res = 1 / (2**config_rpu['bw_inp_res_bits'] - 2)
        rpu.backward.is_perfect = config_rpu['bw_is_perfect']
        rpu.backward.noise_management = eval(config_rpu['bw_noise_management'])
        rpu.backward.out_bound = config_rpu['bw_out_bound']
        rpu.backward.out_noise = config_rpu['bw_out_noise']
        rpu.backward.out_res = 1 / (2**config_rpu['bw_out_res'] - 2)
        rpu.backward.w_noise = config_rpu['bw_w_noise']
        rpu.backward.w_noise_type = eval(config_rpu['bw_w_noise_type'])

    # Parameters - Update
    rpu.update.pulse_type = eval(config_rpu['update_pulse_type'])

    ## Parameters - Device
    if isinstance(rpu, SingleRPUConfig):
        rpu.device.construction_seed = seed
        rpu.device.dw_min = config_rpu['device_dw_min']
        rpu.device.up_down = config_rpu['device_up_down']
        rpu.device.dw_min_dtod = config_rpu['device_dw_min_dtod']
        rpu.device.dw_min_std = config_rpu['device_dw_min_std']
        rpu.device.up_down_dtod = config_rpu['device_up_down_dtod']
        rpu.device.w_max_dtod = config_rpu['device_w_max_dtod']
        rpu.device.w_min_dtod = config_rpu['device_w_min_dtod']


def get_value(parameters, constant_str, param, constant_value):
    """ Get the right value of each parameter """
    count = 0
    for i in parameters:
        index = constant_str.index(i)
        constant_value[index] = param[count]
        count = count + 1
    return constant_value


def rpu_config(device_type, param):
    """ Get the right configuration of the rpu device """
    config_dict = import_config_dict()
    parameters = config_dict['RPU']['parameters']
    # Select Neural Network
    if device_type == 'constant step device':
        if len(param) == 0:
            return SingleRPUConfig(device=ConstantStepDevice())
        # The value of the parameter of the constant step device
        value = [0, 0.0, 1000, 0.0, 0.0, 0.001, 0.3, 0.3, 0.0, 0.0, 0.01, 0.0,
                 0.01, 0.0, 0.01, 0.6, 0.3, -0.6, 0.3]
        # The parameter of the constant step device
        string = ["construction_seed", "corrupt_devices_prob",
                  "corrupt_devices_range", "diffusion", "diffusion_dtod",
                  "dw_min", "dw_min_dtod", "dw_min_std", "lifetime",
                  "lifetime_dtod", "reset", "reset_dtod", "reset_std",
                  "up_down", "up_down_dtod", "w_max", "w_max_dtod",
                  "w_min", "w_min_dtod"]
        value = get_value(parameters, string, param, value)
        rpu = SingleRPUConfig(device=
                              ConstantStepDevice(construction_seed=value[0],
                                                 corrupt_devices_prob=value[1],
                                                 corrupt_devices_range=
                                                 value[2],
                                                 diffusion=value[3],
                                                 diffusion_dtod=value[4],
                                                 dw_min=value[5],
                                                 dw_min_dtod=value[6],
                                                 dw_min_std=value[7],
                                                 lifetime=value[8],
                                                 lifetime_dtod=value[9],
                                                 reset=value[10],
                                                 reset_dtod=value[11],
                                                 reset_std=value[12],
                                                 up_down=value[13],
                                                 up_down_dtod=value[14],
                                                 w_max=value[15],
                                                 w_max_dtod=value[16],
                                                 w_min=value[17],
                                                 w_min_dtod=value[18]))

    elif device_type == 'floating point device':
        rpu = FloatingPointRPUConfig(device=FloatingPointDevice())

    elif device_type == 'ideal device':
        rpu = SingleRPUConfig(device=IdealDevice())

    elif device_type == 'exp step device':
        if len(param) == 0:
            return SingleRPUConfig(device=ExpStepDevice())
        # The value of the parameter of the exp step device
        value = [0, 0.0, 1000, 0.0, 0.0, 0.001, 0.3, 0.3, 0.0, 0.0, 0.01, 0.0,
                 0.01, 0.0, 0.01, 0.6, 0.3, -0.6, 0.3, 0.00081, 0.36833,
                 12.44625, 12.78785, 0.244, 0.2425, 0.0, 0.0, 0.0]
        # The parameter of the exp step device
        string = ["construction_seed", "corrupt_devices_prob",
                  "corrupt_devices_range", "diffusion", "diffusion_dtod",
                  "dw_min", "dw_min_dtod", "dw_min_std", "lifetime",
                  "lifetime_dtod", "reset", "reset_dtod", "reset_std",
                  "up_down", "up_down_dtod", "w_max", "w_max_dtod", "w_min",
                  "w_min_dtod", "A_up", "A_down", "gamma_up", "gamma_down",
                  "a", "b", "dw_min_std_add", "dw_min_std_slope",
                  "write_noise_std"]
        value = get_value(parameters, string, param, value)
        rpu = SingleRPUConfig(device=ExpStepDevice(construction_seed=value[0],
                                                   corrupt_devices_prob=
                                                   value[1],
                                                   corrupt_devices_range=
                                                   value[2],
                                                   diffusion=value[3],
                                                   diffusion_dtod=value[4],
                                                   dw_min=value[5],
                                                   dw_min_dtod=value[6],
                                                   dw_min_std=value[7],
                                                   lifetime=value[8],
                                                   lifetime_dtod=value[9],
                                                   reset=value[10],
                                                   reset_dtod=value[11],
                                                   reset_std=value[12],
                                                   up_down=value[13],
                                                   up_down_dtod=value[14],
                                                   w_max=value[15],
                                                   w_max_dtod=value[16],
                                                   w_min=value[17],
                                                   w_min_dtod=value[18],
                                                   A_up=value[19],
                                                   A_down=value[20],
                                                   gamma_up=value[21],
                                                   gamma_down=value[22],
                                                   a=value[23], b=value[24],
                                                   dw_min_std_add=value[25],
                                                   dw_min_std_slope=value[26],
                                                   write_noise_std=value[27]))

    elif device_type == 'soft bounds device':
        if len(param) == 0:
            return SingleRPUConfig(device=SoftBoundsDevice())
        # The value of the parameter of the soft bounds device
        value = [0, 0.0, 1000, 0.0, 0.0, 0.001, 0.3, 0.3, 0.0, 0.0, 0.01, 0.0,
                 0.01, 0.0, 0.01, 0.6, 0.3, -0.6, 0.3]
        # The parameter of the soft bounds device
        string = ["construction_seed", "corrupt_devices_prob",
                  "corrupt_devices_range", "diffusion", "diffusion_dtod",
                  "dw_min", "dw_min_dtod", "dw_min_std", "lifetime",
                  "lifetime_dtod", "reset", "reset_dtod", "reset_std",
                  "up_down", "up_down_dtod", "w_max", "w_max_dtod", "w_min",
                  "w_min_dtod"]
        value = get_value(parameters, string, param, value)
        rpu = SingleRPUConfig(device=SoftBoundsDevice(construction_seed=
                                                      value[0],
                                                      corrupt_devices_prob=
                                                      value[1],
                                                      corrupt_devices_range=
                                                      value[2],
                                                      diffusion=value[3],
                                                      diffusion_dtod=value[4],
                                                      dw_min=value[5],
                                                      dw_min_dtod=value[6],
                                                      dw_min_std=value[7],
                                                      lifetime=value[8],
                                                      lifetime_dtod=value[9],
                                                      reset=value[10],
                                                      reset_dtod=value[11],
                                                      reset_std=value[12],
                                                      up_down=value[13],
                                                      up_down_dtod=value[14],
                                                      w_max=value[15],
                                                      w_max_dtod=value[16],
                                                      w_min=value[17],
                                                      w_min_dtod=value[18]))

    elif device_type == 'soft bounds pmax device':
        if len(param) == 0:
            return SingleRPUConfig(device=SoftBoundsPmaxDevice())
        # The value of the parameter of the soft bounds pmax device
        value = [0, 0.0, 1000, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 0.01, 0.0, 0.01,
                 0.01, 0.3, 0.3, 1000, 0.0005, -1.0, 1.0]
        # The parameter of the soft bounds pmax device
        string = ["construction_seed", "corrupt_devices_prob",
                  "corrupt_devices_range", "diffusion", "diffusion_dtod",
                  "dw_min_dtod", "dw_min_std", "lifetime", "lifetime_dtod",
                  "reset", "reset_dtod","reset_std", "up_down_dtod",
                  "w_max_dtod", "w_min_dtod", "p_max", "alpha", "range_min",
                  "range_max"]
        if parameters[0] == 'w_max':
            parameters[0] = 'range_max'
        if parameters[1] == 'w_min':
            parameters[1] = 'range_min'
        value = get_value(parameters, string, param, value)
        rpu = SingleRPUConfig(device=
                              SoftBoundsPmaxDevice(construction_seed=value[0],
                                                   corrupt_devices_prob=
                                                   value[1],
                                                   corrupt_devices_range=
                                                   value[2],
                                                   diffusion=value[3],
                                                   diffusion_dtod=value[4],
                                                   dw_min_dtod=value[5],
                                                   dw_min_std=value[6],
                                                   lifetime=value[7],
                                                   lifetime_dtod=value[8],
                                                   reset=value[9],
                                                   reset_dtod=value[10],
                                                   reset_std=value[11],
                                                   up_down_dtod=value[12],
                                                   w_max_dtod=value[13],
                                                   w_min_dtod=value[14],
                                                   p_max=value[15],
                                                   alpha=value[16],
                                                   range_min=value[17],
                                                   range_max=value[18]))

    elif device_type == 'linear step device':
        if len(param) == 0:
            return SingleRPUConfig(device=LinearStepDevice())
        # The value of the parameter of the linear step device
        value = [0, 0.0, 1000, 0.0, 0.0, 0.001, 0.3, 0.3, 0.0, 0.0, 0.01, 0.0,
                 0.01, 0.0, 0.01, 0.6, 0.3, -0.6, 0.3, 0.0, 0.0, 0.05, 0.05,
                 0.0]
        # The parameter of the linear step device
        string = ["construction_seed", "corrupt_devices_prob",
                  "corrupt_devices_range", "diffusion", "diffusion_dtod",
                  "dw_min", "dw_min_dtod", "dw_min_std", "lifetime",
                  "lifetime_dtod", "reset", "reset_dtod", "reset_std",
                  "up_down", "up_down_dtod", "w_max", "w_max_dtod", "w_min",
                  "w_min_dtod", "gamma_up", "gamma_down", "gamma_up_dtod",
                  "gamma_down_dtod", "write_noise_std"]
        value = get_value(parameters, string, param, value)
        rpu = SingleRPUConfig(device=
                              LinearStepDevice(construction_seed=value[0],
                                               corrupt_devices_prob=value[1],
                                               corrupt_devices_range=value[2],
                                               diffusion=value[3],
                                               diffusion_dtod=value[4],
                                               dw_min=value[5],
                                               dw_min_dtod=value[6],
                                               dw_min_std=value[7],
                                               lifetime=value[8],
                                               lifetime_dtod=value[9],
                                               reset=value[10],
                                               reset_dtod=value[11],
                                               reset_std=value[12],
                                               up_down=value[13],
                                               up_down_dtod=value[14],
                                               w_max=value[15],
                                               w_max_dtod=value[16],
                                               w_min=value[17],
                                               w_min_dtod=value[18],
                                               gamma_up=value[19],
                                               gamma_down=value[20],
                                               gamma_up_dtod=value[21],
                                               gamma_down_dtod=value[22],
                                               write_noise_std=value[23]))

    elif device_type == 'pow step device':
        if len(param) == 0:
            return SingleRPUConfig(device=PowStepDevice())
        # The value of the parameter of the pow step device
        value = [0, 0.0, 1000, 0.0, 0.0, 0.001, 0.3, 0.3, 0.0, 0.0, 0.01, 0.0,
                 0.01, 0.0, 0.01, 0.6, 0.3, -0.6, 0.3, 1.0, 0.1, 0.0, 0.0, 0.0]
        # The parameter of the pow step device
        string = ["construction_seed", "corrupt_devices_prob",
                   "corrupt_devices_range", "diffusion", "diffusion_dtod",
                   "dw_min", "dw_min_dtod", "dw_min_std", "lifetime",
                   "lifetime_dtod", "reset", "reset_dtod", "reset_std",
                   "up_down", "up_down_dtod", "w_max", "w_max_dtod", "w_min",
                   "w_min_dtod", "pow_gamma", "pow_gamma_dtod", "pow_up_down",
                   "pow_up_down_dtod", "write_noise_std"]
        value = get_value(parameters, string, param, value)
        rpu = SingleRPUConfig(device=PowStepDevice(construction_seed=value[0],
                                                   corrupt_devices_prob=
                                                   value[1],
                                                   corrupt_devices_range=
                                                   value[2],
                                                   diffusion=value[3],
                                                   diffusion_dtod=value[4],
                                                   dw_min=value[5],
                                                   dw_min_dtod=value[6],
                                                   dw_min_std=value[7],
                                                   lifetime=value[8],
                                                   lifetime_dtod=value[9],
                                                   reset=value[10],
                                                   reset_dtod=value[11],
                                                   reset_std=value[12],
                                                   up_down=value[13],
                                                   up_down_dtod=value[14],
                                                   w_max=value[15],
                                                   w_max_dtod=value[16],
                                                   w_min=value[17],
                                                   w_min_dtod=value[18],
                                                   pow_gamma=value[19],
                                                   pow_gamma_dtod=value[20],
                                                   pow_up_down=value[21],
                                                   pow_up_down_dtod=value[22],
                                                   write_noise_std=value[23]))

    elif device_type == 'hardware aware':
        rpu = InferenceRPUConfig()
        rpu.backward.bound_management = BoundManagementType.NONE
        rpu.forward.out_res = -1.  # Turn off (output) ADC.
        rpu.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
        rpu.forward.w_noise = 0.02
        rpu.noise_model = PCMLikeNoiseModel(g_max=25.0)

    elif device_type == 'hardware aware 2':
        rpu = InferenceRPUConfig()
        rpu.forward.out_res = -1.  # Turn off (output) ADC
        rpu.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
        rpu.forward.w_noise = 0.02  # Short-term w-noise

        rpu.clip.type = WeightClipType.FIXED_VALUE
        rpu.clip.fixed_value = 1.0
        rpu.modifier.pdrop = 0.03  # Drop connect
        # Fwd/bwd weight noise
        rpu.modifier.type = WeightModifierType.ADD_NORMAL
        rpu.modifier.std_dev = 0.1
        rpu.modifier.rel_to_actual_wmax = True

        # Inference noise model
        rpu.noise_model = PCMLikeNoiseModel(g_max=25.0)

        # drift compensation
        rpu.drift_compensation = GlobalDriftCompensation()

    elif device_type == 'hardware aware 3':
        # Inference/hardware-aware training tile
        rpu = InferenceRPUConfig()

        # Specify additional options of the non-idealities in forward pass
        rpu.forward.inp_res = 1 / 64.    # 6-bit DAC discretization
        rpu.forward.out_res = 1 / 256.   # 8-bit ADC discretization
        rpu.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
        rpu.forward.w_noise = 0.02       # Short-term w-noise (weight noise)
        rpu.forward.out_noise = 0.02     # Output noise

        # Specify the noise model to be used for inference only
        rpu.noise_model = PCMLikeNoiseModel(g_max=25.0)  # the model described

        # Specify the drift compensation
        rpu.drift_compensation = GlobalDriftCompensation()

    elif device_type == 'tiki taka CapacitorPresetDevice':
        rpu = TikiTakaCapacitorPreset()

    elif device_type == 'tiki taka EcRamPresetDevice':
        rpu = TikiTakaEcRamPreset()

    elif device_type == 'tiki taka EcRamMOPresetDevice':
        rpu = TikiTakaEcRamMOPreset()

    elif device_type == 'tiki taka IdealizedPresetDevice':
        rpu = TikiTakaIdealizedPreset()

    elif device_type == 'tiki taka ReRamESPresetDevice':
        rpu = TikiTakaReRamESPreset()

    elif device_type == 'tiki taka ReRamSBPresetDevice':
        rpu = TikiTakaReRamSBPreset()

    elif device_type == 'tiki taka':
        # The Tiki-taka learning rule can be implemented by transfer device.
        rpu = UnitCellRPUConfig(
            device=TransferCompound(

                # Devices that compose the Tiki-taka compound.
                unit_cell_devices=[
                    # SoftBoundsDevice(w_min=-0.3, w_max=0.3),
                    # SoftBoundsDevice(w_min=-0.6, w_max=0.6)
                    # FloatingPointDevice(),
                    # FloatingPointDevice
                    ConstantStepDevice(w_min=-10.0, w_max=10.0),
                    ConstantStepDevice(w_min=-10.0, w_max=10.0)

                ],

                # Make some adjustments of the way Tiki-Taka is performed.
                units_in_mbatch=True,  # batch_size=1 anyway
                transfer_every=2,  # every 2 batches do a transfer-read
                n_cols_per_transfer=1,  # one forward read for each transfer
                gamma=0.0,  # all SGD weight in second device
                scale_transfer_lr=True,  # in relative terms to SGD LR
                transfer_lr=1.0,  # same transfer LR as for SGD
            )
        )

        # Make more adjustments (can be made here or above).
        ## rpu.forward.inp_res = 1 / 64.  # 6 bit DAC

        # same backward pass settings as forward
        ## rpu.backward = rpu.forward

        # Same forward/update for transfer-read as for actual SGD.
        rpu.device.transfer_forward = rpu.forward

        # SGD update/transfer-update will be done with stochastic pulsing.
        rpu.device.transfer_update = rpu.update

    elif device_type == 'mixed precision':
        # https://www.frontiersin.org/articles/10.3389/fnins.2020.00406/full
        rpu = DigitalRankUpdateRPUConfig(
            device=MixedPrecisionCompound(
                # device=SoftBoundsDevice(),
                device=ConstantStepDevice(w_min=-10.0, w_max=10.0),

                # make some adjustments of mixed-precision hyperparameter
                granularity=0.0,
                n_x_bins=0,  # floating point activations for Chi update
                n_d_bins=0,  # floating point delta for Chi update
            )
        )

    elif device_type == 'mixed precision CapacitorPreset':
        rpu = MixedPrecisionCapacitorPreset()

    elif device_type == 'mixed precision ECRamPreset':
        rpu = MixedPrecisionEcRamPreset()

    elif device_type == 'mixed precision EcRamMOPreset':
        rpu = MixedPrecisionEcRamMOPreset()

    elif device_type == 'mixed precision GokmenVlasovPreset':
        rpu = MixedPrecisionGokmenVlasovPreset()

    elif device_type == 'mixed precision IdealizedPreset':
        rpu = MixedPrecisionIdealizedPreset()

    elif device_type == 'mixed precision PCMPreset':
        rpu = MixedPrecisionPCMPreset()

    elif device_type == 'mixed precision ReRamESPreset':
        rpu = MixedPrecisionReRamESPreset()

    elif device_type == 'mixed precision ReRamSBPreset':
        rpu = MixedPrecisionReRamSBPreset()

    return rpu
