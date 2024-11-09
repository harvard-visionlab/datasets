from PIL import Image
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import CenterCropRGBImageDecoder, SimpleRGBImageDecoder, IntDecoder
# from ffcv.transforms import ToTensor, Normalize
from ffcv.pipeline.operation import Operation
import numpy as np
from ffcv.fields import IntField, RGBImageField

from .custom_decoders import SimpleSampleReader, get_max_sample_size

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

default_custom_fields = {
    'image': RGBImageField,
    'label': IntField,
}

class FFCVDataset:
    def __init__(self, beton_file, pipelines=None, custom_fields=default_custom_fields, 
                 image_field='image', label_field='label'):
        """
        Initialize a pseudo-dataset from a .beton file for random access to individual samples.
        
        DO NOT USE FOR MODEL TRAINING OR EVALUATION.
        
        Why does this exist then? Because sometimes it's nice to be able to randomly access 
        samples and inspect them. Or to just get metadata about a dataset.
        
        Args:
            beton_file (str): Path to the .beton file.
            image_pipeline (list): List of transformations for images.
            label_pipeline (list): List of transformations for labels.
        """
        self.beton_file = beton_file
        self.image_field = image_field
        self.label_field = image_field
        max_size = get_max_sample_size(beton_file, custom_fields=custom_fields)
        self.pipelines = pipelines if pipelines is not None else {
            image_field: [SimpleSampleReader(max_size)],
            # image_field: [CenterCropRGBImageDecoder((224,224), DEFAULT_CROP_RATIO)],
            label_field: [IntDecoder()],
        }
        self.custom_fields = custom_fields
        
        self.loader = Loader(
            path=self.beton_file,
            batch_size=1,
            num_workers=1,
            order=OrderOption.SEQUENTIAL,
            pipelines=self.pipelines,
            custom_fields=self.custom_fields,
        )
    
    def set_index(self, index):
        """Set the loader to access a specific single index."""
        # Update indices to the desired single index
        self.loader.indices = np.array([index], dtype="uint64")
        self.loader._args['indices'] = self.loader.indices.tolist()
        
        # Reinitialize traversal order and pipeline if needed
        self.loader.traversal_order = self.loader.traversal_order.__class__(self.loader)
        self.loader.generate_code()  # Recompile code with updated indices
        
    def __getitem__(self, index):
        """Access an individual sample by index."""
        self.set_index(index)
        
        # Retrieve the sample from the loader
        sample = [values[0] for values in next(iter(self.loader))]
        
        return sample
    
    def __len__(self):
        """Get the number of samples in the dataset."""
        return self.loader.reader.num_samples
    
    def __repr__(self) -> str:
        _repr_indent = 4
        tab = " " * _repr_indent
        head = "FFCVDataset (use only for viewing samples; use FFCVLoader directly for training/eval)"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"beton_file: {self.beton_file}")
        body.append(f"field_names: {self.loader.reader.field_names}")
        body.append(f"field_name_to_f_ix: {self.loader.field_name_to_f_ix}")
        
        body += [f"\nPipelines:"]
        for k,transforms in self.pipelines.items():
            body += [f"{k}:{[v.__class__.__name__ for v in transforms]}"]
            
        lines = [head] + [" " * _repr_indent + line for line in body]
        
        lines += ['\nSample Info:']
        sample = self.__getitem__(0)
        any_bytes = False
        if hasattr(sample, 'items'):
            for key,value in sample.items():
                lines += [" " * _repr_indent + f"{key}: {type(value)}"]
                if isinstance(value, bytes): 
                    any_bytes = True
        else:
            for idx,value in enumerate(sample):
                lines += [" " * _repr_indent + f"{idx}: {type(value)}"]
                if isinstance(value, bytes): 
                    any_bytes = True            
                    
        if self.image_field is not None:
            image_index = self.loader.field_name_to_f_ix[self.image_field]
            lines += ['\nUsage Examples:']
            lines += [f'{tab}import io']
            lines += [f'{tab}from PIL import Image']
            lines += [f"{tab}sample = dataset[0]"]
            
            if isinstance(self.pipelines[self.image_field][0], SimpleSampleReader):
                lines += [f"{tab}pil_image = Image.open(io.BytesIO(sample[{image_index}]))"]
            else:
                lines += [f"{tab}pil_image = Image.fromarray(sample[{image_index}])"]
                
        return "\n".join(lines)