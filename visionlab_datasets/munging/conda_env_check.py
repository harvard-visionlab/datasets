if __name__ == "__main__":    
    import requests
    from io import BytesIO
    import ffcv
    import cv2
    import torch
    # import composer
    # import streaming
    import PIL
    import PIL.features
    from PIL import Image
    import numpy as np
    import timm
    import litdata
    # from streaming import MDSWriter, StreamingDataset

    print(f'ffcv: {ffcv.__version__}')
    print(f'cv2: {cv2.__version__}')
    print(f'PIL: {PIL.__version__}, simd={".post"  in PIL.__version__}, turbo={PIL.features.check_feature("libjpeg_turbo")}')
    print(f'torch: {torch.__version__}, cuda_is_available={torch.cuda.is_available()}')
    print(f'torch hub directory: {torch.hub.get_dir()}')
    print(f'timm version: {timm.__version__}')
    print(f'litdata version: {litdata.__version__}')
    # print(f'composer: {composer.__version__}')
    # print(f'streaming: {streaming.__version__}')
    print("\n")

    url = 'https://www.dropbox.com/scl/fi/vfj60rucpsl5d6ednkrcc/dog.jpg?rlkey=loej8dbrwdvjsw6xafrp7smj4&dl=1'
    print(f'loading: {url}')
    response = requests.get(url)
    img_data = BytesIO(response.content)
    img = Image.open(img_data)
    img.save('/tmp/dog.jpg')
    
    filename = '/tmp/dog.jpg'
    print(f'loading: {filename}')
    img = Image.open(filename)
    img = np.array(img)
    print(img.shape)
    print(img[0])