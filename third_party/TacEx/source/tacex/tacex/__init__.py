"""
Python module for simulating GelSight sensors inside Isaac Sim/Lab
"""

from .gelsight_sensor import GelSightSensor
from .gelsight_sensor_cfg import GelSightSensorCfg
from .gelsight_sensor_data import GelSightSensorData

# Register UI extensions (only when omni.ui is available — skip in headless mode).
try:
    from .ui_extension_example import UsdrtExamplePythonExtension
    _HAS_UI = True
except ImportError:
    UsdrtExamplePythonExtension = None
    _HAS_UI = False

__all__ = ["GelSightSensor", "GelSightSensorCfg", "GelSightSensorData"]
if _HAS_UI:
    __all__.append("UsdrtExamplePythonExtension")
