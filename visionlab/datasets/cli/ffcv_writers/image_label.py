__all__ = ['image_label']

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField
from .custom_fields import RGBImageField

def image_label(
    write_path, 
    write_mode: str = 'jpg', 
    smart_threshold: int = None,
    min_resolution: int = None, 
    max_resolution: int = None,
    max_enforced_with: str = 'center_crop',
    compress_probability: float = 0.5,
    jpeg_quality: int = 100,
    num_workers: int = 12,
):
    
    fields = {
        'image': RGBImageField(write_mode=write_mode,
                               smart_threshold=smart_threshold,
                               min_resolution=min_resolution,
                               max_resolution=max_resolution,
                               max_enforced_with=max_enforced_with,
                               compress_probability=compress_probability,
                               jpeg_quality=jpeg_quality),
        'label': IntField(),
    }
    
    writer = DatasetWriter(write_path, fields, num_workers=num_workers)  
    
    return writer