from numpy.lib.shape_base import expand_dims
import module.loss_fn as loss_fn
import module.unet as unet
import torch
import torch.nn as nn
from torch import optim
import wandb
import os
import cv2
import numpy as np
from tqdm import tqdm

class NN(nn.Module):
    def __init__(self, args):
        super(NN, self).__init__()
        print('build networks')
        self.backbone = unet.UNet(args.num_channels, args.num_classes)
        self.criterion = loss_fn.DepthLoss(args)
        self.optimizer = optim.Adam(self.backbone.parameters(), lr=args.lr)
        print('Done.')

    def train_step(self, batch, args):
        imgs = batch['image'].to(device=torch.device('cuda'))
        gt = batch['label'].to(device=torch.device('cuda'))
        param_depth = args.depth
        param_grad = args.grad
        param_norm = args.norm
        if args.nonlinear:
            losses = self.criterion((args.nonlinear_c*self.backbone(imgs)).sigmoid(), gt)
        else:
            losses = self.criterion(self.backbone(imgs), gt)
        loss = param_depth*losses[0] + param_grad*losses[1] + param_norm*losses[2]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return losses, loss

    def train_epoch(self, dataset, epoch, args):
        self.backbone.train()        
        avg_loss = 0
        dataset = tqdm(dataset, desc=f'Epoch: {epoch+1}')
        for batch in dataset:
            losses, loss = self.train_step(batch, args)
            avg_loss += loss/len(dataset)
            dataset.set_postfix({
                'depth loss': '{0:1.5f}'.format(losses[0].item()),
                'grad loss': '{0:1.5f}'.format(losses[1].item()),
                'norm loss': '{0:1.5f}'.format(losses[2].item())
                })
            wandb.log({
                'depth loss': losses[0].item(),
                'grad loss': losses[1].item(),
                'norm loss': losses[2].item()
            })
        dataset.close()
        return avg_loss
    
    @torch.no_grad()
    def verbose(self, img, mask, epoch, args):
        '''
        Input:
            img: C*H*W tensor
            mask: H*W int array 
        Output:
            None
        '''
        mask = mask.squeeze()
        verbose_url = args.save_dir + '/verbose'
        verbose_img = (img.cpu().numpy()*255).astype('uint8')
        img_channels = img.shape[0]
        # convert img to rgb
        if img_channels == 1:
            verbose_img = cv2.cvtColor(verbose_img.transpose(1, 2, 0), cv2.COLOR_GRAY2RGB)
        elif img_channels == 2:
            zeros = np.zeros_like(verbose_img[0:1])
            verbose_img = np.concatenate([verbose_img, zeros]).transpose(1, 2, 0)

        # save verbose img
        try:
            os.makedirs(verbose_url)
        except:
            pass
        if not os.path.exists(verbose_url+'/input.jpg'):
            cv2.imwrite(verbose_url+'/input.jpg', verbose_img)
        
        canvas = verbose_img
        canvas[mask>0, 0] = 255
        # save mask
        output = cv2.addWeighted(verbose_img, 0.8, canvas, 0.2, 1)
        cv2.imwrite(verbose_url+f'/epoch_{epoch+1}.jpg', output)

    @torch.no_grad()
    def eval(self, dataset, epoch, args):
        self.backbone.eval()
        avg_loss = 0
        outputs = []
        metric = None
        masks = None
        if args.verbose:
            verbose_flag = True
        
        dataset = tqdm(dataset, desc=f'Eval Epoch: {epoch+1}')
        for batch in dataset:
            imgs = batch['image'].to(device=torch.device('cuda'))
            gt = batch['label'].to(device=torch.device('cuda'))
            if args.nonlinear:
                pred = (args.nonlinear_c*self.backbone(imgs)).sigmoid()
            else:
                pred = self.backbone(imgs)
            loss = torch.abs(gt - pred).mean()
            avg_loss += loss.item() / len(dataset)
            # loss = self.criterion(pred, gt)
            flow = self.criterion.edge_conv(pred)
            flow = flow / torch.norm(flow, dim=1, keepdim=True)
            flow[(pred<0.01).repeat(1, 2, 1, 1)] = 0
            outputs.append(flow.cpu())
            dataset.set_postfix({
                'loss': '{0:1.5f}'.format(loss.item())
                })
            if verbose_flag:
                verbose_img = imgs[0]
                verbose_flag = False
        outputs = torch.cat(outputs)
        
        if not args.mode == 'test':
            # if args.verbose:
            #     mask = dataset.iterable.dataset.label_to_annotation(outputs[0:1])[:, 1].astype(int)
            #     self.verbose(verbose_img, mask, epoch, args)
            wandb.log({'eval loss': avg_loss, "epoch":epoch})
            if (epoch+1) % args.save_interval == 0:
                torch.save({
                    'model_state_dict': self.backbone.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join(args.save_dir, f'epoch_{epoch+1}.pth'))  
                # metric, masks = dataset.iterable.dataset.metric(outputs, args, verbose=True)
                # metric_mean = np.mean(metric, axis=0)
                # wandb.log({
                #     'val AP50': metric_mean[0],
                #     'val AP75': metric_mean[1],
                #     'val mAP': metric_mean[2]
                #     })
        else:
            metric, masks = dataset.iterable.dataset.metric(outputs, args, verbose=True)
        
        dataset.close()
        
        return metric, masks

