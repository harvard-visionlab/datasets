import os
import contextlib
import scipy.io as sio

from ...auth import get_aws_credentials
from ...utils import download_s3_file
from ...datasets import StreamingDatasetVisionlab
from ...registry import dataset_registry

s3_urls = {
    "STIMULI": "s3://visionlab-litdata/exploring-objects-images/",
    "SECTORS": "s3://visionlab-brain-data/konkle_72objects/rawdata/ExploringObjectsData_SECTORS.mat",
    "GRADIENT": "s3://visionlab-brain-data/konkle_72objects/rawdata/ExploringObjectsData_GRADIENT.mat",
}

__all__ = ['NeuralDatasetSectors', 'NeuralDatasetGradient', 'StimulusDataset']

class NeuralDataset():
    def __init__(self, dataset="SECTORS", profile_name='default'):
        self.dataset = dataset
        self.remote_file = s3_urls[dataset]
        # Suppress print output from download_s3_file
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            self.local_file = download_s3_file(self.remote_file, dryrun=False, profile_name=profile_name)

        self.data = sio.loadmat(self.local_file, simplify_cells=True)[self.dataset]
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __repr__(self):
        lines = []
        tab = " " * 4
        lines += [f"{self.__class__.__name__}"]
        lines += [f"{tab}local_file: {self.local_file}"]
        lines += [f"{tab}remote_file: {self.remote_file}"]
                
        lines += [f"\nData Fields:"]
        lines += [f"{tab}{list(self.data.keys())}"]
        
        lines += [f"\nROIs:"]
        lines += [f"{tab}{list(self.data['Betas'].keys())}"]
        
        lines += ["\nUsage:"]
        lines += [f"{tab}data = dataset[field_name]"]
        lines += [f"{tab}conditions = dataset['conditions']"]
        lines += [f"{tab}betas = dataset['Betas']"]
        
        return "\n".join(lines)
    
@dataset_registry.register("neuro", "konkle_72_objects", "SECTORS", "72 Inanimate Objects fMRI Betas by SECTOR (Konkle, )")
class NeuralDatasetGradient(NeuralDataset):
    def __init__(self, profile_name='default'):
        super().__init__(dataset="GRADIENT", profile_name=profile_name)

@dataset_registry.register("neuro", "konkle_72_objects", "GRADIENT", "72 Inanimate Objects fMRI Betas by GRADIENT ROI (Konkle, )")
class NeuralDatasetSectors(NeuralDataset):
    def __init__(self, profile_name='default'):
        super().__init__(dataset="SECTORS", profile_name=profile_name)    

@dataset_registry.register("neuro", "konkle_72_objects", "stimuli", "72 Inanimate Object Stimuli (PIL Images)")
class StimulusDataset(StreamingDatasetVisionlab):
    def __init__(self, profile_name='default', **kwargs):
        storage_options = get_aws_credentials(profile_name)
        super().__init__(s3_urls['STIMULI'], storage_options=storage_options, **kwargs)