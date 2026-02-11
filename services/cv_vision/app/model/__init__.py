# Model package
from .encoder import VisionTransformerEncoder, CopernicusViTEncoder
from .copernicus_encoder import CopernicusFMEncoder, CopernicusFMMultiTaskModel
from .classification_head import ClassificationDecoder
from .detection_head import ObjectDetectionDecoder
from .segmentation_head import SegmentationDecoder
from .copernicus_segmentation import CopernicusFMSegmentation, CopernicusFMSegmentationWrapper
from .upernet_decoder import UPerNetDecoder, SimpleUPerHead, SimpleFCNHead
from .integrated_model import MultiTaskModel, MultiTaskWrapper

# Backward compatibility
from .integrated_model import MultiTaskWrapper as DualTaskWrapper
from .integrated_model import MultiTaskModel as DualTaskModel

__all__ = [
    'VisionTransformerEncoder',
    'CopernicusViTEncoder',
    'CopernicusFMEncoder',
    'CopernicusFMMultiTaskModel',
    'ClassificationDecoder', 
    'ObjectDetectionDecoder',
    'SegmentationDecoder',
    'CopernicusFMSegmentation',
    'CopernicusFMSegmentationWrapper',
    'UPerNetDecoder',
    'SimpleUPerHead',
    'SimpleFCNHead',
    'MultiTaskModel',
    'MultiTaskWrapper',
    'DualTaskWrapper',  # Backward compatibility
    'DualTaskModel'     # Backward compatibility
]
