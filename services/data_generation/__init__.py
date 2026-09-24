"""
Sensor Augmentation Engine Package
==================================

Physics-based synthetic sensor and urban context generation models for
Smart Road Digital Twin.
"""

from .lidar_model import LiDARModel
from .sonar_model import SonarModel
from .accelerometer_model import AccelerometerModel
from .traffic_model import TrafficModel
from .infrastructure_model import InfrastructureModel
from .location_model import (
    LocationModel,
    MODE_SOURCE_LOCATION,
    MODE_SYNTHETIC_LOCATION,
    MODE_EXTERNAL_GIS_LOCATION
)
from .consistency_model import PhysicalConsistencyModel
from .sensor_augmentation import SensorAugmentationEngine

__all__ = [
    "LiDARModel",
    "SonarModel",
    "AccelerometerModel",
    "TrafficModel",
    "InfrastructureModel",
    "LocationModel",
    "MODE_SOURCE_LOCATION",
    "MODE_SYNTHETIC_LOCATION",
    "MODE_EXTERNAL_GIS_LOCATION",
    "PhysicalConsistencyModel",
    "SensorAugmentationEngine"
]
