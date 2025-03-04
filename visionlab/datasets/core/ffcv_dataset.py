from PIL import Image
from ffcv.reader import Reader
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import CenterCropRGBImageDecoder, SimpleRGBImageDecoder, IntDecoder, BytesDecoder
# from ffcv.transforms import ToTensor, Normalize
from ffcv.pipeline.operation import Operation
import numpy as np
from ffcv.fields import IntField, RGBImageField
from ffcv.utils import decode_null_terminated_string
from ffcv.types import (ALLOC_TABLE_TYPE, HeaderType, CURRENT_VERSION,
                    FieldDescType, get_handlers, get_metadata_type)
from pdb import set_trace

from .custom_decoders import SimpleSampleReader, get_max_sample_size

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

default_custom_fields = {
    'image': RGBImageField,
    'label': IntField,
}

class FFCVDataset:
    def __init__(self, beton_file, pipelines={}, custom_fields=None, 
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
        self.metadata = self.get_ffcv_metadata(beton_file)
        self.image_field = image_field
        self.label_field = label_field
        if custom_fields is None:
            self.custom_fields = {name: RGBImageField for name in self.metadata['field_names'] if image_field in name}
        else:
            self.custom_fields = custom_fields
        
        max_size = get_max_sample_size(beton_file, custom_fields=self.custom_fields)
        
        for field_name in self.metadata['field_names']:
            if field_name in pipelines: 
                continue
            if image_field in field_name:
                pipelines[field_name] = [SimpleSampleReader(max_size)]
            elif field_name in ['label', 'index']:
                pipelines[field_name] = [IntDecoder()]
            else:
                pipelines[field_name] = [BytesDecoder()]

        self.pipelines = pipelines
        print("wtf", self.pipelines)
        print(custom_fields)
        
        self.loader = Loader(
            path=self.beton_file,
            batch_size=1,
            num_workers=1,
            order=OrderOption.SEQUENTIAL,
            pipelines=self.pipelines,
            custom_fields=self.custom_fields,
        )
    
    def get_ffcv_metadata(self, _path):
    
        # get header info
        header = np.fromfile(_path, dtype=HeaderType, count=1)[0]
        header.setflags(write=False)
        version = header["version"]

        if version != CURRENT_VERSION:
            msg = f"file format mismatch: code={CURRENT_VERSION},file={version}"
            raise AssertionError(msg)

        num_samples = header["num_samples"]
        page_size = header["page_size"]
        num_fields = header["num_fields"]

        # get field names
        offset = HeaderType.itemsize
        field_descriptors = np.fromfile(
            _path, dtype=FieldDescType, count=num_fields, offset=offset
        )
        field_descriptors.setflags(write=False)
        handlers = get_handlers(field_descriptors)

        field_descriptors = field_descriptors
        field_names = list(
            map(decode_null_terminated_string, field_descriptors["name"])
        )

        return dict(num_samples=num_samples, field_names=field_names)
    
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
            body += [f"{k}:{[v.__class__.__name__ for v in transforms] if transforms is not None else 'None'}"]
            
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
