import cv2
import numpy as np

def plot_mask(images, masks, gt_masks, url):
    for i, (mask, gt_mask) in enumerate(zip(masks, gt_masks)):
        img = images[i]
        img_channels = img.shape[0]
        if img_channels == 1:
            img = cv2.cvtColor((img.transpose(1, 2, 0)*255).astype('uint8'), cv2.COLOR_GRAY2RGB)
        elif img_channels == 2:
            zeros = np.zeros_like(img[0:1])
            img = np.concatenate([img, zeros]).transpose(1, 2, 0)
        canvas = img
        mask_ = img
        mask_[mask>0, 2] = 255
        mask_[gt_mask>0, 1] = 255
        canvas = cv2.addWeighted(canvas, 0.8, mask_, 0.2, 0)
        cv2.imwrite(url + f'/{i}.jpg', canvas)