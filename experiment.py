import wandb
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
import dataset
from args import describe
import module

#### save exp info ###
args = describe('test')

### prepare experiment Material ###
if args.aux:
    net = module.Aux_Net(args).cuda()
else:       
    net = module.NN(args).cuda()
  
if args.mode == 'train':
    wandb.init(dir=args.save_dir, config=args)
    train_set = dataset.load_train_dataset(args)
    val_set = dataset.load_val_dataset(args)
    #debug
    # net.backbone.load_state_dict(torch.load('result/01_18_19_50_58/epoch_10.pth')['model_state_dict'])
    ### baseline ###
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, num_workers=4, batch_size=args.batch_size)
    for e in np.arange(args.epochs):
        # net.train_epoch(train_loader, e, args)
        stats, masks = net.eval(val_loader, e, args)
    
    # test_set = dataset.load_test_dataset(args)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size)
    # stats, masks = net.eval(test_loader, 0, args)
    # print(np.mean(stats, axis=0))
    # np.savetxt(args.save_dir+'/performance.txt', stats)
### test ###

if args.mode == 'test':
    test_set = dataset.load_test_dataset(args)
    test_loader = DataLoader(test_set, batch_size=12)
    state_dict = torch.load(args.nn_path)
    if args.aux:
        net.backbone.load_state_dict(state_dict['unet_state_dict'])
        net.aux_net.load_state_dict(state_dict['aux_state_dict'])
    else:
        net.backbone.load_state_dict(state_dict['model_state_dict'])
    stats, masks = net.eval(test_loader, 0, args)
    print(np.mean(stats, axis=0))
    np.savetxt(args.save_dir+'/performance.txt', stats)
    worst_case = np.argsort(stats[:, -1])
