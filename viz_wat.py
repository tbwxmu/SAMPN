from argparse import ArgumentParser, Namespace
import os, random
import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_checkpoint
from cv import  DGLDataset, DGLCollator

from tqdm import tqdm
import numpy as np
from cv import evaluate_batch
from utils import get_task_names, get_metric_func
from scaler import StandardScaler, minmaxScaler

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
    metric_func = get_metric_func(metric=args.metric)

    for it, result_batch in enumerate(tqdm(viz_dataloader)):


        batch_sm = result_batch['sm']
        label_batch=result_batch['labels']
        if args.dataset_type == 'regression':
            if args.scale=="standardization":
                print('Fitting scaler(Z-score standardization)')
                scaler = StandardScaler().fit(label_batch)
                y_train_scaled = scaler.transform(label_batch)
                print(f'train data mean:{scaler.means}\nstd:{scaler.stds}\n')
            if args.scale=="normalization":
                print('Fitting scaler( Min-Max normalization )')
                scaler = minmaxScaler().fit(label_batch)
                y_train_scaled = scaler.transform(label_batch)
                print(f'train data min:{scaler.mins}\ntrain data max:{scaler.maxs}\n')
            if args.scale !='standardization' and args.scale!='normalization':
                raise ValueError("not implemented scaler,use one of [standardization, normalization]")
        else:
            scaler = None
        mpn.viz_attention(batch_sm, viz_dir=args.viz_dir)
        test_targets,test_preds,test_scores = evaluate_batch(args,
                            model=model,
                            data=viz_dataloader,
                            num_tasks=args.num_tasks,
                            metric_func=metric_func,
                            dataset_type=args.dataset_type,
                            scaler=scaler,
                            logger=None,
                            Foldth=0,
                            predsLog=args.save_dir)
        print(f'rung viz{args.viz_dir}')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data_RE2/tmp/0_testcc',
                        help='Path to data CSV file')
    parser.add_argument('--viz_dir', type=str, default='viz_attention',
                        help='Path where attention PNGs will be saved')
    parser.add_argument('--checkpoint_path', type=str, default='save_test/fold_0/model_0/model.pt',
                        help='Path to a model checkpoint')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    args = parser.parse_args()


    args.seed=seed
    seed_torch(seed)
    args.sumstyle=True



    args.data_path='data_RE2/ol_wat.csv'
    args.data_filename=os.path.basename(args.data_path)+f'_seed{args.seed}'


    args.viz_dir='png_seed3032wat_ol'
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    args.checkpoint_path='save_test/fold_0/model_0/water_solubilityOCD.csv_seed3032_model.pt'
    args.batch_size=128
    args.attention=True
    args.dataset_type='regression'
    args.scale="normalization"
    args.num_tasks=1
    args.metric='rmse'
    args.save_dir='save_test'
    del args.no_cuda

    os.makedirs(args.viz_dir, exist_ok=True)
    print(f'args:\t-->{args}')

    visualize_attention(args)




