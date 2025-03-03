import torch
import json

class JSONFileDataset():
    def __init__(self, local_file, fields=[], **kwargs):
        self.local_file = local_file     
        self.fields = fields
        self.load_data(**kwargs)
        
    def load_data(self, **kwargs):
        with open(self.local_file, "r") as file:
            data = json.load(file)
        
        if self.fields:
            data = {field: data[field] for field in self.fields}

        self.data = data            
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __repr__(self):
        lines = []
        tab = " " * 4
        lines += [f"{self.__class__.__name__}"]
        lines += [f"{tab}local_file: {self.local_file}"]
          
        num_items = len(self.data.items())
        lines += [f"\nData Fields (N={num_items}):"]
        for count,(k,v) in enumerate(self.data.items()):
            lines += [f"{tab}{k}: {type(v)}"]
            if count > 10 and count/num_items < .50:
                lines += [f"{tab}...{num_items-count} other fields"]
                break
                
        lines += ["\nUsage:"]
        lines += [f"{tab}data = dataset[field_name]"]
        # lines += [f"{tab}betas = dataset['Betas']"]
        
        return "\n".join(lines)