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
        self.pose_url = os.path.split(label_url)[0] + '/pose_label'
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
            inds = np.array(np.nonzero(np.abs(prediction[0])>0)).astype(np.int32).T
            p_start = torch.stack(torch.meshgrid(torch.arange(predictions.shape[-2]), torch.arange(predictions.shape[-1])))
            p_start = p_start.cuda()
            dP = prediction.cuda()[[1, 0]]
            p_end, inds = metric.follow_flows_gpu(-dP, p_start, inds=inds, niter=2000)
            maski = metric.get_masks(p_end.cpu().numpy(), iscell=None, flows=None)
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
        # import matplotlib.pyplot as plt
        # pred_url = args.save_dir + '/pred'
        # os.makedirs(pred_url, exist_ok=True)
        # for i, (pred, label) in enumerate(zip(prediction, self.labels)):
        #     plt.figure(figsize=(16, 6), dpi=200)
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(pred[0], cmap='plasma')
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(label, cmap='plasma')
        #     plt.savefig(pred_url+f'/{i}.png')
        #     break
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
        depth = self.labels[index]
        
        if len(image.shape) == 2:
            image = image[None, :, :]
        if len(depth.shape) == 2:
            depth = depth[None, :, :]
        flow_url = os.path.join(self.pose_url, f'{index}.npy')
        flow = np.load(flow_url)[1:]
        if len(flow.shape) == 4:
            flow = flow[0]
        label = np.concatenate([depth, flow], axis=0)
        image, label = self.crop(image, label)
        return {
            'image': torch.FloatTensor(image),
            'label': torch.FloatTensor(label)
        }     
