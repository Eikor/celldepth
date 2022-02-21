import argparse
import time
import os
import yaml

def parse():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--epochs', type=int, default=100, help='epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for main task')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size for training')
    parser.add_argument('--save_interval', type=int, default=10)
    
    # net
    parser.add_argument('--aux', action='store_true')
    parser.add_argument('--aux_channels', type=int, default=8)

    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--depth', type=float, default=1)
    parser.add_argument('--grad', type=float, default=1)
    parser.add_argument('--norm', type=float, default=1)
    parser.add_argument('--nonlinear', action='store_true')
    parser.add_argument('--nonlinear_c', type=float, default=1)

    # save dir
    parser.add_argument('--save_dir', type=str, default='./result', help='directory to save record file')

    # dataset
    parser.add_argument('--dataset', type=str, default='livecell')
    parser.add_argument('--crop_size', type=int, default=320)

    
    ### test args
    parser.add_argument('--test_url', type=str, default=None)

    return parser.parse_args()

def get_args(mode):
    num_channels = {
        'livecell': 1,
        'tissuenet': 2,
    }
    segmentation = ['pose', 'softpose']
    detection = ['point']
    args = parse()
    args.mode = mode

    args.train_image_url = f'./dataset/data/{args.dataset}/train/train_images.npy'
    args.train_anno_url = f'./dataset/data/{args.dataset}/train/train_annotation.npy'
    args.train_label_url = f'./dataset/data/{args.dataset}/train/train_label.npy'
    args.test_image_url = f'./dataset/data/{args.dataset}/test/test_images.npy'
    args.test_anno_url = f'./dataset/data/{args.dataset}/test/test_annotation.npy'
    args.test_label_url = f'./dataset/data/{args.dataset}/test/test_label.npy'
    args.val_image_url = f'./dataset/data/{args.dataset}/val/val_images.npy'
    args.val_anno_url = f'./dataset/data/{args.dataset}/val/val_annotation.npy'
    args.val_label_url = f'./dataset/data/{args.dataset}/val/val_label.npy'
    
    args.num_channels = num_channels[args.dataset]
    

    args.verbose = True
    if args.mode == 'test':
        assert args.test_url is not None
        args.save_dir = args.test_url
        args.nn_path = os.path.join(args.save_dir, 'epoch_80.pth')
        with open(args.save_dir+'/wandb/latest-run/files/config.yaml', 'r') as f:
            settings = yaml.load(f.read())
            args.aux = settings['aux']['value']
            args.aux_channels = settings['aux_channels']['value']
            args.nonlinear = settings['nonlinear']['value']
            args.nonlinear_c = settings['nonlinear_c']['value']
    else:
        args.save_dir = 'result/' + time.strftime('%m_%d_%H_%M_%S',time.localtime(int(round(time.time()))))
    try:
        os.makedirs(args.save_dir)
        print(f'save experiment result in {args.save_dir}')
    except:
        print(f'save experiment result in {args.save_dir}')
    return args

def describe(mode):
    args = get_args(mode)
    # args.mode = mode
    return args