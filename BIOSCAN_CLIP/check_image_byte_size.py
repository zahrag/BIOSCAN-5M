import io
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def pil_image_to_byte(img):
    binary_data_io = io.BytesIO()
    img.save(binary_data_io, format='JPEG')
    binary_data = binary_data_io.getvalue()
    curr_image_np = np.frombuffer(binary_data, dtype=np.uint8)
    return curr_image_np

if __name__ == '__main__':
    max_size = 0

    special_arg_image_dir = '/localhome/zmgong/second_ssd/data/BIOSCAN_6M_cropped/organized/cropped_resized'
    for root, dirs, files in os.walk(special_arg_image_dir):
        for file in tqdm(files):
            if file.endswith('.jpg'):
                img = Image.open(os.path.join(root, file))
                img_np = pil_image_to_byte(img)
                if img_np.size > max_size:
                    max_size = img_np.size
    print(f'Maximum image size: {max_size}')
