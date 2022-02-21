import torch
import torch.nn as nn
import numpy as np

class AuxLoss(nn.Module):
    def __init__(self, args):
        super(AuxLoss, self).__init__()
        self.main = FlowLoss()
        self.aux = DepthLoss()
   
    def forward(self, y_hat, y, reduction='sum', masked=False):
        '''
            y: tensor with shape [b*1*H*W]
                [
                    [depth]
                    ...
                ]
        '''

        loss_flow = self.main(y_hat, y)
        loss_depth = self.aux(y_hat, y)
        return loss_depth + loss_flow

class DepthLoss(nn.Module):
    def __init__(self, args):
        super(DepthLoss, self).__init__()
        self.main = nn.L1Loss(reduction='mean')
        # self.aux = nn.MSELoss(reduction='mean')
        # self.cos = nn.CosineSimilarity()
        # self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        # edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        # edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # edge_k = np.stack((edge_ky, edge_kx))

        # edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        # self.edge_conv.weight = nn.Parameter(edge_k)
        
        # for param in self.parameters():
        #     param.requires_grad = False

   
    def forward(self, y_hat, y, reduction='sum', masked=False):
        '''
            y: tensor with shape [b*1*H*W]
                [
                    [depth]
                    ...
                ]
        '''
        if y.shape[1] > 1:
            gt_depth = y[:, 0:1, :, :]
        else:
            gt_depth = y
        if y_hat.shape[1] > 1:
            pred_depth = y_hat[:, 0:1, :, :]
        else:
            pred_depth = y_hat
        # grad = self.edge_conv(y_hat)
        # gt_grad = self.edge_conv(y)
        # norm = torch.cat((-grad, torch.ones_like(grad[:, 0:1, :, :])), 1)
        # gt_norm = torch.cat((-gt_grad, torch.ones_like(grad[:, 0:1, :, :])), 1)
        # mask = torch.where(y>0, y+0.01, torch.ones_like(y))
        loss_depth = self.main(pred_depth, gt_depth)
        # loss_grad = torch.log(torch.abs(grad - gt_grad) + 0.5).mean()
        # loss_normal = torch.abs(1 - self.cos(norm, gt_norm)).mean()

        return loss_depth

class FlowLoss(nn.Module):
    def __init__(self, args):
        super(FlowLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_ky, edge_kx))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

    def forward(self, y_hat, y, reduce='mean'):
        if y.shape[1] > 1:
            gt_grad = y[:, 1:, :, :]
        else:
            gt_grad = self.edge_conv(y)
        if y_hat.shape[1] > 1:
            if y_hat.shape[1] == 2:
                pred_grad = y_hat
        else:
            pred_grad = self.edge_conv(y_hat)
        loss = self.l1(pred_grad, gt_grad)
        return loss

