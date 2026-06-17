# from __future__ import annotations
from isaaclab.utils import configclass

from ..vbts_simulator_cfg import VBTSSimulatorCfg
from .mlp_fots_sim import MLPFOTSSimulator

@configclass
class MLPFOTSSimulatorCfg(VBTSSimulatorCfg):
    """
    Configuration class for MLP-FOTS optical simulation approach.
    """

    simulation_approach_class: type = MLPFOTSSimulator

    calib_folder_path: str = ""
    """Path to calibration folder."""

    device: str = "cuda"

    tactile_img_res: tuple = (480, 480)
    """Resolution of the tactile image output (width, height)."""

    mlp_weights_filename: str = "mlp_n2c_vitai5.pth"
    """Filename of MLP weights in calib folder."""

    bg_img_filename: str = "vitai_bg.npy"
    """Filename of background RGB image in calib folder."""


