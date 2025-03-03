import cv2
import numpy as np
from PIL import Image

from pdb import set_trace
    
__all__ = ['turbo_loader', 'pil_loader', 'cv2_loader', 'open_image']

try:
    # https://github.com/lilohuang/PyTurboJPEG/blob/master/turbojpeg.py
    from turbojpeg import TurboJPEG, TJPF_RGB, TJPF_BGR
except:
    # https://github.com/loopbio/PyTurboJPEG/tree/loopbio
    from turbojpeg import TurboJPEG, TJPF
    TJPF_RGB = TJPF.RGB
    TJPF_BGR = TJPF.BGR
    
turbo = TurboJPEG()

def turbo_loader(file, to_rgb=True):
    with open(file, 'rb') as f:
        # have to install latest to access crop features:
        # buf = turbo.crop(f.read(), x=0, y=0, w=100, h=100, preserve=False, gray=True)
        img = turbo.decode(f.read(), pixel_format=TJPF_RGB if to_rgb else TJPF_BGR)
    return img

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def cv2_loader(path, to_rgb=True):
    img = cv2.imread(path)
    if to_rgb: img = img[:,:,::-1]
    
    return img

def open_image(p, to_rgb=True):
    '''Our default image loader, takes `filename` and returns a PIL Image. 
        Speedwise, turbo_loader > pil_loader > cv2, but cv2 is the most robust, so 
        we try to load jpg images with turbo_loader, fall back to PIL, then cv2.
        
        This fallback behavior is needed, e.g., with ImageNet there are a few images
        that either aren't JPEGs or have issues that turbo_loader crashes on, but cv2 
        doesn't.
    '''
    if p.lower().endswith('.jpg') or p.lower().endswith('.jpeg'): 
        try:
            img = turbo_loader(p, to_rgb=to_rgb)
        except:
            try:
                img = pil_loader(p)
            except:
                img = cv2.imread(p)
                if to_rgb: img = img[:,:,::-1]
    else:
        try:
            img = pil_loader(p)
        except:
            img = cv2.imread(p)
            if to_rgb: img = img[:,:,::-1]
                
    if img is not None and not isinstance(img, Image.Image):
        img = Image.fromarray(img)
        
    return img