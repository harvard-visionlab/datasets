from torchvision import transforms
from PIL import Image

# ===============================================================
#  Transforms
# ===============================================================

class ConvertToRGB:
    """Convert the given image to RGB format."""

    def __call__(self, img):
        return img.convert('RGB')

    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
class ResizeShortWithMaxLong():
    def __init__(self, short_size, longest_size):
        self.short_size = short_size
        self.longest_size = longest_size

        self.resize = transforms.Resize(short_size)
        self.crop_width = transforms.CenterCrop( (short_size, longest_size) )
        self.crop_height = transforms.CenterCrop( (longest_size, short_size) )

    def __call__(self, img):
        img = self.resize(img)

        # Check the image dimensions
        width, height = img.size

        # If width exceeds longest_size, crop width
        if width > self.longest_size:
            img = self.crop_width(img)

        # If height exceeds longest_size, crop height
        if height > self.longest_size:
            img = self.crop_height(img)

        width, height = img.size
        assert width==self.short_size or height==self.short_size, f"Oops, one side should equal {self.short_size}"
        assert width <= self.longest_size, f"Oops, width exceeds {self.longest_size}, {width}"
        assert height <= self.longest_size, f"Oops, height exceeds {self.longest_size}, {height}"

        return img