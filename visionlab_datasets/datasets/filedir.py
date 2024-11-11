import os
from pathlib import Path
from pdb import set_trace

from .source_types import get_source_format, DatasetFormat
from .streaming_dataset import StreamingDatasetVisionlab
from .ffcv_dataset import FFCVDataset
from .matfile import MatFileDataset
from .pytorchfile import PyTorchFileDataset
from .subfolder import SubFolderDataset
from ..utils.archive_helpers import decompress_if_needed

dataset_classes = {
    DatasetFormat.STREAMING: StreamingDatasetVisionlab,
    DatasetFormat.FFCV: FFCVDataset,
    DatasetFormat.IMAGE_DIR: SubFolderDataset,
    DatasetFormat.MAT: MatFileDataset,
    DatasetFormat.PTH: PyTorchFileDataset,
}

class FileDirDataset():
    def __init__(self, local_files):
        if isinstance(local_files, str) and os.path.isdir(local_files):
            local_files = self.get_file_list(local_files)
        local_paths = [decompress_if_needed(local_file, ignore_non_archives=True) for local_file in local_files]
        self.local_paths = {Path(local_path).name:local_path for local_path in local_paths}
        # self.load_data(**kwargs)
    
    def get_file_list(self, root_dir):
        file_list = []
        for root, dirs, fnames in os.walk(root_dir):
            dirs.clear()
            for fname in fnames:
                file_list.append(os.path.join(root, fname))
        return file_list
                                 
    def load_data(self, **kwargs):
        rawdata = sio.loadmat(self.local_file, simplify_cells=self.simplify_cells, **kwargs)
        data = {}
        if self.fields:
            data = {field: rawdata[field] for field in self.fields}
        else:
            data = {field: rawdata[field] for field in rawdata.keys() if not field.startswith("__")}
        
        self.data = data            
        
    def __getitem__(self, index, **kwargs):
        source = get_source_format(self.local_paths[index])
        datset_cls = dataset_classes[type(source)]
        dataset = datset_cls(source, **kwargs)
        return dataset
    
    def load_dataset(self, index, **kwargs):
        return self.__getitem__(index, **kwargs)
    
    def __repr__(self):
        lines = []
        tab = " " * 4
        lines += [f"{self.__class__.__name__}"]
        lines += [f"{tab}subsets: {list(self.local_paths.keys())}"]
            
        for key,local_path in self.local_paths.items():
            local_path = get_source_format(local_path)
            lines += [' ']
            lines += [f"{tab}{key}"]
            lines += [f"{tab}{tab}type: {type(local_path).__name__}"]
            lines += [f"{tab}{tab}path: {local_path}"]
     
        lines += ["\nUsage:"]
        lines += [f"{tab}subset = dataset[subset_name]"]
        
        for key in self.local_paths.keys():
            lines += [f"{tab}subset = dataset['{key}']"]
        
        return "\n".join(lines)