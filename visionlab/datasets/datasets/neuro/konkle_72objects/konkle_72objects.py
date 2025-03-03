import os
import contextlib
import scipy.io as sio
from botocore import UNSIGNED
from botocore.client import Config

from ....utils import get_aws_credentials, fetch
from ....core import StreamingDatasetVisionlab
from ....registry import dataset_registry

from pdb import set_trace

s3_urls = {
    "STIMULI": "s3://visionlab-litdata/exploring-objects-images/",
    "SECTORS": "https://s3.amazonaws.com/visionlab-brain-data/konkle_72objects/rawdata/ExploringObjectsData_SECTORS.mat",
    "GRADIENT": "https://s3.amazonaws.com/visionlab-brain-data/konkle_72objects/rawdata/ExploringObjectsData_GRADIENT.mat",
}

__all__ = ['NeuralDatasetSectors', 'NeuralDatasetGradient', 'StimulusDataset']

class NeuralDataset():
    def __init__(self, dataset="SECTORS", profile_name=None):        
        self.dataset = dataset
        self.remote_file = s3_urls[dataset]
        # Suppress print output from download_s3_file
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            self.local_file = fetch(self.remote_file, dryrun=False, profile_name=profile_name)
        
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
    
@dataset_registry.register("neuro", "konkle_72_objects", split="SECTORS", fmt='matfile', metadata="72 Inanimate Objects fMRI Betas by SECTOR (Konkle, )")
class NeuralDatasetSectors(NeuralDataset):
    def __init__(self, profile_name=None):
        super().__init__(dataset="SECTORS", profile_name=profile_name)

@dataset_registry.register("neuro", "konkle_72_objects", split="GRADIENT", fmt='matfile', metadata="72 Inanimate Objects fMRI Betas by GRADIENT ROI (Konkle, )")
class NeuralDatasetGradient(NeuralDataset):
    def __init__(self, profile_name=None):
        super().__init__(dataset="GRADIENT", profile_name=profile_name)    

@dataset_registry.register("neuro", "konkle_72_objects", split="stimuli", fmt='images', metadata="72 Inanimate Object Stimuli (PIL Images)")
class StimulusDataset(StreamingDatasetVisionlab):
    def __init__(self, profile_name=None, **kwargs):
        storage_options = get_aws_credentials(profile_name)
        storage_options['endpoint_url']="https://s3.amazonaws.com"
        storage_options['config'] = Config(signature_version=UNSIGNED)
        super().__init__(s3_urls['STIMULI'], storage_options=storage_options, **kwargs)