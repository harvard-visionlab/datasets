import torch

class PyTorchFileDataset():
    def __init__(self, local_file, fields=[], map_location='cpu', weights_only=False, **kwargs):
        self.local_file = local_file     
        self.fields = fields
        self.map_location = map_location
        self.weights_only = weights_only
        self.load_data(**kwargs)
        
    def load_data(self, **kwargs):
        rawdata = torch.load(self.local_file, 
                             map_location=self.map_location, 
                             weights_only=self.weights_only, 
                             **kwargs)
        data = {}
        if self.fields:
            data = {field: rawdata[field] for field in self.fields}
        else:
            data = {field: rawdata[field] for field in rawdata.keys() if not field.startswith("__")}
        
        self.data = data            
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __repr__(self):
        lines = []
        tab = " " * 4
        lines += [f"{self.__class__.__name__}"]
        lines += [f"{tab}local_file: {self.local_file}"]
                
        lines += [f"\nData Fields:"]
        for k,v in self.data.items():
            lines += [f"{tab}{k}: {type(v)}"]
     
        lines += ["\nUsage:"]
        lines += [f"{tab}data = dataset[field_name]"]
        # lines += [f"{tab}betas = dataset['Betas']"]
        
        return "\n".join(lines)