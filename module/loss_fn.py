import torch
import torch.nn as nn

class DepthLoss(nn.Module):
    def __init__(self, args):
        super(DepthLoss, self).__init__()
        self.main = nn.L1Loss(reduction='mean')
        self.aux = nn.MSELoss(reduction='mean')
        self.cos = nn.CosineSimilarity()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

   
    def forward(self, y_hat, y, reduction='sum', masked=False):
        '''
            y: tensor with shape [b*1*H*W]
                [
                    [depth]
                    ...
                ]
        '''
        grad = self.edge_conv(y_hat)
        gt_grad = self.edge_conv(y)
        norm = torch.cat((-grad, torch.ones_like(grad[:, 0:1, :, :])), 1)
        gt_norm = torch.cat((-gt_grad, torch.ones_like(grad[:, 0:1, :, :])), 1)
        loss_depth = torch.abs(y_hat - y) / y
        loss_grad = torch.log(torch.abs(grad - gt_grad) + 0.5).mean()
        loss_normal = torch.abs(1 - self.cos(norm, gt_norm)).mean()

        return [loss_depth, loss_grad, loss_normal]

class PointLoss(nn.Module):
    def __init__(self, args):
        super(PointLoss, self).__init__()

    def forward(self, y_hat, y):
        prob = torch.sigmoid(y_hat).clamp(min=1e-4, max=(1 - 1e-4))
        loss = torch.mean(torch.log(prob) * (y==1) + torch.log(1 - prob) * (1-y))
        return -loss

    # @torch.no_grad()
    # def eval(self, y_hat):
    #     prob = torch.sigmoid(y_hat).clamp(min=1e-4, max=(1 - 1e-4))
    #     return {'output': prob.detach().cpu().numpy()}          

class PoseLoss(nn.Module):
    def __init__(self, args):
        super(PoseLoss, self).__init__()
        self.alpha = args.pose_alpha
        self.beta = args.pose_beta
        self.nonlinear_flow = args.nonlinear_flow
        self.c = args.flow_c
        self.thresh = 0.5
        if args.pose_loss == 'l1':
            self.l2 = nn.L1Loss(reduction='none')
        else:
            self.l2 = nn.MSELoss(reduction='none')
   
    def forward(self, y_hat, y, reduction='sum', masked=False):
        '''
        weights: 
            1 for cell；
            (0, 1) for low-confidence cell;
            0 for unlabeled pixel, 50-50
            [-1, 0) for background
        '''
        prob = torch.sigmoid(y_hat[:, 0:1, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
        flow = y_hat[:, 1:, :, :]
        weights = y[:, 0:1, :, :]
        gt_flow = y[:, 1:, :, :]
        
        pos_mask = weights > self.thresh
        if torch.any(weights<0):
            neg_mask = weights < 0
        else: neg_mask = weights == 0
        select_mask = pos_mask + neg_mask

        prob_loss = -torch.sum(torch.log(prob) * pos_mask + torch.log(1 - prob) * (neg_mask)) / torch.sum(select_mask)
        if self.nonlinear_flow:
            flow_loss = self.l2(torch.tanh(self.c * flow), gt_flow)
        else:
            flow_loss = self.l2(flow, gt_flow)
        
        if masked:
            if torch.sum(pos_mask) == 0:
                flow_loss = torch.sum(flow_loss * pos_mask)
            else:
                flow_loss = torch.sum(flow_loss* pos_mask) / torch.sum(pos_mask)
        else:
            flow_loss = torch.sum(flow_loss * select_mask) / torch.sum(select_mask)

        return self.beta * flow_loss + self.alpha * prob_loss
    
    # @torch.no_grad()
    # def eval(self, y_hat):
    #     prob = torch.sigmoid(y_hat[:, 0, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
    #     flow = y_hat[:, 1:, :, :]
    #     return {
    #         'output': prob.detach().cpy().numpy(),
    #         'flow': flow.detach().cpu().numpy()
    #     }

class MaskLoss(nn.Module):
    def __init__(self, args):
        super(MaskLoss, self).__init__()
        self.alpha = args.pose_alpha
        self.beta = args.pose_beta
        self.thresh = 0.5
   
    def forward(self, y_hat, y, reduce='mean'):
        '''
        weights: 
            1 for cell；
            (0, 1) for low-confidence cell;
            0 for unlabeled pixel, 50-50
            [-1, 0) for background
        '''
        pos_mask = y == 1
        neg_mask = y == 0
        prob = torch.sigmoid(y_hat[:, 0, :, :]).clamp(min=1e-4, max=(1 - 1e-4))
        prob_loss = torch.log(prob) * pos_mask + torch.log(1 - prob) * neg_mask
        if reduce == 'mean':
            return - torch.sum(prob_loss) / torch.sum(pos_mask + neg_mask)
        else:
            return -prob_loss    

class FlowLoss(nn.Module):
    def __init__(self, args):
        super(FlowLoss, self).__init__()
        self.beta = args.pose_beta
        self.l2 = nn.L1Loss(reduction='none')
   
    def forward(self, y_hat, y, reduce='mean'):
        flow = y_hat[:, 1:, :, :]
        weights = y[:, 0:1, :, :]
        gt_flow = y[:, 1:, :, :]
        
        pos_mask = weights == 1
        loss = self.beta * self.l2(flow, gt_flow) * pos_mask
        if reduce == 'mean':
            if torch.sum(pos_mask) == 0:
                return torch.sum(loss)
            return torch.sum(loss) / torch.sum(pos_mask)
        else:
            return loss

class ConsistLoss(nn.Module):
    def __init__(self, args):
        super(ConsistLoss, self).__init__()
        self.consist = args.consist
        self.l2 = nn.MSELoss(reduce=None)
   
    def forward(self, y_f, y_g, reduce='mean'):
        flow_f = y_f[:, 1:, :, :]
        flow_g = y_g[:, 1:, :, :]

        loss = self.consist * self.l2(flow_f, flow_g)
        if reduce == 'mean':
            return torch.mean(loss)
        else:
            return loss

import torch.nn.functional as F
import math
import numpy as np
from scipy.special import lambertw

class SuperLoss(nn.Module):

    def __init__(self, C=10, lam=1, batch_size=128):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size
                  
    def forward(self, logits, targets):
        l_i = F.cross_entropy(logits, targets, reduction='none').detach()
        sigma = self.sigma(l_i)
        loss = (F.l1_loss(logits, targets, reduction='none') - self.tau)*sigma + self.lam*(torch.log(sigma)**2)
        loss = loss.sum()/self.batch_size
        return loss

    def sigma(self, l_i):
        x = torch.ones(l_i.size())*(-2/math.exp(1.))
        x = x.cuda()
        y = 0.5*torch.max(x, (l_i-self.tau)/self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma

class SoftPoseLoss(nn.Module):
    def __init__(self, args):
        super(SoftPoseLoss, self).__init__()
        self.a = args.soft_a
        self.b = args.soft_b
        self.c = args.soft_c
        self.nonlinear_flow = args.nonlinear_flow
        self.c = args.flow_c
        self.thresh = 0.5
        self.soft_mask = args.soft_mask
        self.main = nn.L1Loss(reduction='none')
        self.aux = nn.MSELoss(reduction='mean')
   
    def forward(self, y_hat, y, reduction='sum', masked=False):
        '''
            y: tensor with shape [b*4*H*W]
                [
                    [point, boundary, vflow, hflow]
                    ...
                ]
        '''
        center = y_hat[:, 0:1, :, :]
        boundary = y_hat[:, 1:2, :, :]
        flow = y_hat[:, 2:, :, :]
        gt_center = y[:, 0:1, :, :]
        gt_boundary = y[:, 1:2, :, :]
        gt_flow = y[:, 2:, :, :]
        
        center_loss = self.aux(center, gt_center)
        boundary_loss = self.aux(boundary, gt_boundary)
        
        soft_mask = torch.clamp(1 - gt_center - gt_boundary, min=0, max=1)
        if self.nonlinear_flow:
            flow_loss = self.main(torch.tanh(self.c * flow), gt_flow)
        else:
            flow_loss = self.main(flow, gt_flow)
        
        if self.soft_mask:
            flow_loss = flow_loss * soft_mask

        sum_loss = self.a*center_loss + self.b*boundary_loss + self.c*torch.mean(flow_loss)
        return [center_loss, boundary_loss, torch.mean(flow_loss), sum_loss]