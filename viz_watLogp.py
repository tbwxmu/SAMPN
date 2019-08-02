from argparse import ArgumentParser, Namespace
import os, random
import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_checkpoint
from cv import  DGLDataset, DGLCollator

from tqdm import tqdm
import numpy as np

seed=3032
def seed_torch(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    print(f'used seed in seed_torch={seed}<+++++++++++')
def worker_init_fn():


    np.random.seed(seed)


def visualize_attention(args: Namespace):
    """Visualizes attention weights."""
    print(f'Loading model from "{args.checkpoint_path}"')
    model = load_checkpoint(args.checkpoint_path, cuda=args.cuda)
    mpn = model.encoder
    print(f'mpn:-->{type(mpn)}')
    print(f'MPNencoder attributes:{mpn.encoder.__dict__}')
    print('Loading data')
    if os.path.exists(args.data_path) and os.path.getsize(args.data_path) > 0:
        DGLtest=args.data_path
        print(f'Loading data -->{DGLtest}')
    else:
        direct = 'data_RE2/tmp/'
        DGLtest=direct+'viz.csv'
        print(f'Loading data -->{DGLtest}')

    viz_data=DGLDataset(DGLtest,training=False)
    viz_dataloader = DataLoader(viz_data, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             collate_fn=DGLCollator(training=False),
                             drop_last=False,
                             worker_init_fn=worker_init_fn)

    for it, result_batch in enumerate(tqdm(viz_dataloader)):

        batch_sm = result_batch['sm']
        label_batch=result_batch['labels']
        mpn.viz_attention(batch_sm, viz_dir=args.viz_dir)
        print(f'rung viz{it}')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data_RE2/tmp/0_testcc',
                        help='Path to data CSV file')
    parser.add_argument('--viz_dir', type=str, default='viz_attention2',
                        help='Path where attention PNGs will be saved')
    parser.add_argument('--checkpoint_path', type=str, default='save_test/fold_0/model_0/model.pt',
                        help='Path to a model checkpoint')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    args = parser.parse_args()


    seed_torch(seed)
    args.sumstyle=True
    args.data_path='data_RE2/no_null.csv'



    args.viz_dir='png_seed3032_waterLogp_ol'
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.checkpoint_path='save_test/fold_0/model_0/wat_logP.csvAll_cols.csv_seed3032_model.pt'
    args.batch_size=128
    args.attention=True

    del args.no_cuda

    os.makedirs(args.viz_dir, exist_ok=True)
    print(f'args:\t-->{args}')

    visualize_attention(args)




