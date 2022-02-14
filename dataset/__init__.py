from dataset.celldepth import Depth
from torch.utils.data import DataLoader, random_split


def load_train_dataset(args):
    print(f'load {args.dataset}')
    print('-----train-----')
    train_set = Depth(args.train_image_url, args.train_anno_url, args.train_label_url, args)
    print('-------------')
    return train_set

def load_val_dataset(args):
    print(f'load {args.dataset}')
    print('-----val-----')
    val_set = Depth(args.val_image_url, args.val_anno_url, args.val_label_url, args, Aug=False)
    print('-------------')
    return val_set

def load_test_dataset(args):
    print(f'load {args.dataset}')
    print('-----test-----')
    test_set = Depth(args.test_image_url, args.test_anno_url, args.test_label_url, args, Aug=False)
    print('-------------')
    return test_set