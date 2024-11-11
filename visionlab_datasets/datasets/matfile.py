import scipy.io as sio

class MatFileDataset():
    def __init__(self, local_file, fields=[], simplify_cells=True, **kwargs):
        self.local_file = local_file     
        self.fields = fields
        self.simplify_cells = simplify_cells
        self.load_data(**kwargs)
        
    def load_data(self, **kwargs):
        rawdata = sio.loadmat(self.local_file, simplify_cells=self.simplify_cells, **kwargs)
        data = {}
        if self.fields:
            data = {field: rawdata[field] for field in self.fields}
        else:
            data = {field: rawdata[field] for field in rawdata.keys() if not field.startswith("__")}
        
        keys = list(data.keys())
        if self.fields and len(keys)==1:
            self.data = data[keys[0]]
        else:
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