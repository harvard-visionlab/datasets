from litdata import StreamingDataset
from litdata.utilities.dataset_utilities import _read_updated_at
from pdb import set_trace

class StreamingDatasetVisionlab(StreamingDataset):
    def __init__(self, *args, transforms=None, image_field='image', expected_version=None, **kwargs):
        self.transforms = transforms
        self.image_field = image_field        
        super().__init__(*args, **kwargs)
        self.version = _read_updated_at(self.input_dir) if self.input_dir is not None else None
        if expected_version is not None:
            assert self.version == str(expected_version), f"\n==> expected_version={expected_version}, got dataset.version={self.version}"
            
    def __getitem__(self, idx):
        sample  = super().__getitem__(idx) # <- Whatever you returned from the DatasetOptimizer prepare_item method.
        if self.transforms is not None:
            if self.image_field not in sample:
                fields = list(sample.keys())
                raise ValueError(f"To use transforms, specify 'image_field', value '{self.image_field}' not in sample: {fields}")
            sample[self.image_field] = self.transforms(sample[self.image_field])
        return sample
    
    def __repr__(self) -> str:
        _repr_indent = 4
        tab = " " * _repr_indent
        head = "StreamingDataset"
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.input_dir is not None:
            body.append(f"local_dir: {self.input_dir.path}")
            body.append(f"remote_dir: {self.input_dir.url}")
            body.append(f"version: {self.version}")            
            
        body += [f"\nTransforms on '{self.image_field}':"]
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)
                     .replace("\n",f"\n{tab}")
                     .replace('\n)', '\n' + ' ' * _repr_indent + ')')]
        else:
            body += ['None']
            
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
                    
        if any_bytes:
            lines += ['\nUsage Examples:']
            lines += [f'{tab}import io']
            lines += [f'{tab}from PIL import Image']
            lines += [f"{tab}sample = dataset[0]"]
            
            if hasattr(sample, 'items'):
                for key,value in sample.items():
                    if isinstance(value, bytes):
                        lines += [f"{tab}pil_image = Image.open(io.BytesIO(sample['{key}']))"]
            else:
                lines += [f"{tab}pil_image = Image.open(io.BytesIO(sample[0]))"]
            return "\n".join(lines)