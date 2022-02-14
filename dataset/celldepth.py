import numpy as np
from dataset.utils import metric, plot
from torch.utils.data import Dataset
import torch
import cv2 as cv
from tqdm import tqdm
import os


class Depth(Dataset):
    def __init__(self, image_url, annotation_url, label_url, args, Aug=True) -> None:
        super().__init__()
        print('build depth dataset')
        self.annotations = np.load(annotation_url)
        self.labels = np.load(label_url)
        self.images = np.load(image_url)

        self.CROP_SIZE = 320
        if not Aug:
            self.CROP_SIZE = -1
        self.iou_thresh = 0.5
        print('Done.')
    
    def __len__(self):
        return len(self.images)

    def label_to_annotation(self, predictions, mode='get'):
        '''
        input:
            predictions: array with shape n*1*H*W
                [
                    image_1: [cell_depth]
                    image_2: ...
                    ...
                ]
        output:
        i    annotations: n*H*W
            [
                image_1: [mask]
                image_2: ...
                ... 
            ]
        '''
        annotations = []
        for prediction in tqdm(predictions, desc='simulating'):
            prediction = prediction[0].numpy()
            x_flow = cv.Sobel(prediction, cv.CV_64F, 1, 0)
            y_flow = cv.Sobel(prediction, cv.CV_64F, 0, 1)
            norm = np.linalg.norm(np.stack([x_flow, y_flow]), axis=0)
            norm = np.where(norm==0, 1, norm) 
            dP = np.stack([y_flow, x_flow]) / norm
            dP = np.where(prediction<=0, 0, dP)
            p = metric.follow_flows(-dP, niter=200)
            maski = metric.get_masks(p, iscell=None, flows=None)
            maski = metric.fill_holes_and_remove_small_masks(maski, min_size=80)
            annotations.append(maski)
        return np.array(annotations)

    def metric(self, prediction, args, verbose=False):
        '''
        metric top 100 prediction
        input:
            prediction: array with shape n*3*H*W
        output:
            stats: [precision, recall, mean error]
            masks : n*H*W
            [
                mask1, 
                mask2, 
                ...
            ]
        '''
        masks = self.label_to_annotation(prediction).astype(int)
        gt_masks = self.annotations
        print('calculate iou')
        '''
        '''
        stats = metric.metric(masks, gt_masks, 0.5)
        if verbose:
            test_url = args.save_dir + '/test'
            os.makedirs(test_url, exist_ok=True)
            plot.plot_mask(self.images, masks, gt_masks, test_url)
        import matplotlib.pyplot as plt
        pred_url = args.save_dir + '/pred'
        os.makedirs(pred_url, exist_ok=True)
        for i, (pred, label) in enumerate(zip(prediction, self.labels)):
            plt.figure(figsize=(16, 6), dpi=200)
            plt.subplot(1, 2, 1)
            plt.imshow(pred[0], cmap='plasma')
            plt.subplot(1, 2, 2)
            plt.imshow(label, cmap='plasma')
            plt.savefig(pred_url+f'/{i}.png')
            break
        stats = np.array(stats)
        return stats, masks

    def crop(self, img, label):
        if self.CROP_SIZE < 0:
            return img, label
        if len(label.shape)==4:
            label = label.squeeze()
        crop_size = self.CROP_SIZE
        h, w = img.shape[1:]     
        # random crop images
        tl = np.array([np.random.randint(0, h-crop_size),
                    np.random.randint(0, w-crop_size)])
        br = tl + crop_size

        ### crop image ###
        img_patch, label_patch = img[:, tl[0]:br[0], tl[1]:br[1]], label[:, tl[0]:br[0], tl[1]:br[1]]
        
        return img_patch, label_patch

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if len(image.shape) == 2:
            image = image[None, :, :]
        if len(label.shape) == 2:
            label = label[None, :, :]
        image, label = self.crop(image, label)
        return {
            'image': torch.FloatTensor(image),
            'label': torch.FloatTensor(label)
        }     
