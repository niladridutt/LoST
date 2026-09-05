import rembg
import numpy as np
from PIL import Image
from torchvision import transforms as T


def crop_recenter(image_no_bg, thereshold=100):
    image_no_bg_np = np.array(image_no_bg)
    if image_no_bg_np.shape[2] == 3:
        return image_no_bg
    mask = (image_no_bg_np[..., -1]).astype(np.uint8)
    mask_bin = mask > thereshold
    
    H, W = image_no_bg_np.shape[:2]
    
    valid_pixels = mask_bin.astype(np.float32).nonzero()
    if np.sum(mask_bin) < (H*W) * 0.001:
        min_h = 0
        max_h = H - 1
        min_w = 0
        max_w = W -1
    else:
        min_h, max_h = valid_pixels[0].min(), valid_pixels[0].max()
        min_w, max_w = valid_pixels[1].min(), valid_pixels[1].max()
    
    if min_h < 0:
        min_h = 0
    if min_w < 0:
        min_w = 0
    if max_h > H:
        max_h = H - 1
    if max_w > W:
        max_w = W - 1

    image_no_bg_np = image_no_bg_np[min_h:max_h+1, min_w:max_w+1]
    image_no_bg = Image.fromarray(image_no_bg_np)
    return image_no_bg


def pad_to_same_size(image, pad_value=1):
    image = np.array(image)
    h, w, _ = image.shape
    image_temp = image.copy()
    if h != w:
        # find the max one and pad the other side with white
        max_size = max(h, w)
        
        pad_h = max_size - h
        pad_w = max_size - w
        pad_h_top = max(pad_h // 2, 0)
        pad_h_bottom = max(pad_h - pad_h_top, 0)
        pad_w_left = max(pad_w // 2, 0)
        pad_w_right = max(pad_w - pad_w_left, 0)
        
        image_temp = np.pad(
            image[..., :3], 
            ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)),
            constant_values=pad_value
        )
        if image.shape[2] == 4:
            image_bg = np.pad(
                image[..., 3:], 
                ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)),
                constant_values=0
            )
            image = np.concatenate([image_temp, image_bg], axis=2)
        else:
            image = image_temp
        
    return Image.fromarray(image)


def remove_bg(image):
    image = rembg.remove(image)
    return image


def preprocess(image, rmbg=True):

    if rmbg:
        image = remove_bg(image)

    image = crop_recenter(image)
    image = pad_to_same_size(image, pad_value=255)
    image = np.array(image)
    image = image / 255.
    if image.shape[2] == 4:
        image = image[..., :3] * image[..., 3:] + (1 - image[..., 3:])
    image = Image.fromarray((image * 255).astype('uint8'), "RGB")

    W, H = image.size[:2]
    pad_margin = int(W * 0.04)
    image_transforms = T.Compose([
        T.Pad((pad_margin, pad_margin, pad_margin, pad_margin), fill=255),
        T.ToTensor(),
    ])

    image = image_transforms(image)

    return image
    