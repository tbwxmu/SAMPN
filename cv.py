from sklearn.model_selection import KFold, train_test_split
from logging import Logger
import logging
from argparse import Namespace
from typing import Callable, List,Tuple

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os, csv, random
from pprint import pformat
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tensorboardX import SummaryWriter
from utils import create_logger, get_task_names, build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint
from DGLmodels import build_model, QSARmodel
from scaler import StandardScaler, minmaxScaler
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from nn_utils import param_count, compute_gnorm, compute_pnorm, NoamLR, move_to_cuda, move_dgl_to_cuda
import  math
from sklearn.metrics import mean_absolute_error, mean_squared_error,  r2_score
from sklearn.metrics import auc,roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, roc_curve, accuracy_score

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union

import matplotlib.pyplot as plt
def train_valdation_curve(args,hold_loss,hold_avgVal,fold_i=None, model_idx=None,cur_name='train-valdation curve'):

    fig, ax = plt.subplots()
    plt.plot(np.array(hold_loss),label="Training")
    plt.plot(np.array(hold_avgVal),label="Valdation")
    ax.yaxis.grid(True)
    ax.set_title(f'{cur_name}')
    plt.ylabel('RMSE')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig(f'{args.data_filename}_{fold_i}_{model_idx}_{cur_name}.png', dpi=300, format='png', bbox_inches='tight')

def acc_score(targets: List[int], preds: List[float]) -> float:
    bin_preds = [1 if p > 0.5 else 0 for p in preds]
    return accuracy_score(targets, bin_preds)
def prec_score(targets: List[int], preds: List[float]) -> float:
    bin_preds = [1 if p > 0.5 else 0 for p in preds]
    return precision_score(targets, bin_preds)
def prec_rec_auc(targets: List[int], preds: List[float]) -> float:
    precision, recall, _ = precision_recall_curve(targets, preds)

    return auc(recall, precision)
def rec_score(targets: List[int], preds: List[float]) -> float:
    bin_preds = [1 if p > 0.5 else 0 for p in preds]
    return recall_score(targets, bin_preds)


def rmse(targets: List[float], preds: List[float]) -> float:
    """ Computes the root mean squared error.is the root of mse:mean_squared_error"""
    return math.sqrt(mean_squared_error(targets, preds))

def Pearson_cor(targets: List[float], preds: List[float]) -> float:
    return np.corrcoef(targets,preds)[0, 1]



def get_class_sizes(data):
    targets = data.targets()


    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:

        assert set(np.unique(task_targets)) <= {0, 1}

        try:
            ones = np.count_nonzero(task_targets) / len(task_targets)

        except ZeroDivisionError:
            ones = float('nan')

        class_sizes.append([1 - ones, ones])

    return class_sizes

def generate_scaffold(smiles, include_chirality=False):
  """Compute the Bemis-Murcko scaffold for a SMILES string."""
  from rdkit import Chem
  mol = Chem.MolFromSmiles(smiles)
  engine = ScaffoldGenerator(include_chirality=include_chirality)
  scaffold = engine.get_scaffold(mol)
  return scaffold

class ScaffoldGenerator(object):
  def __init__(self, include_chirality=False):
    self.include_chirality = include_chirality

  def get_scaffold(self, mol):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=self.include_chirality)

def scaffold_to_smiles(all_smiles: List[str], use_indices: bool = True) -> Dict[str, Union[Set[str], Set[int]]]:
    scaffolds = defaultdict(set)
    for i, smiles in tqdm(enumerate(all_smiles), total=len(all_smiles)):
        scaffold = generate_scaffold(smiles)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(smiles)
    return scaffolds

def scaffold_split(data,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   seed: int = 0,
                   balanced: bool = False,
                   scaff_random: bool=False,
                   big_small: int=2,
                   use_indices: bool=True,
                   logger: logging.Logger = None):
    assert sum(sizes) == 1

    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    print(f'the splited size train val tset{train_size},{val_size}，{test_size}')
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    scaffold_to_indices = scaffold_to_smiles(data, use_indices=use_indices)
    if balanced:
        print(f'Put stuff that is bigger than half the val/test size into train, rest just order randomly')
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for scaff,index_set in scaffold_to_indices.items():
            if len(index_set) > val_size / big_small or len(index_set) > test_size / big_small:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)

        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    if scaff_random:
        print(f' scaffold sets are random ordered')
        idx = list(scaffold_to_indices.keys())

        random.shuffle(idx)
        index_sets = [scaffold_to_indices[id] for id in idx]
    else:
        print(f'Sort from largest to smallest scaffold sets')
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) > train_size:
            if len(train)+ len(val) + len(index_set) > val_size+train_size:
                test += index_set
                test_scaffold_count += 1
            else:
                val += index_set
                val_scaffold_count += 1
        else:
            train += index_set
            train_scaffold_count += 1
    print('train val test with',len(train),len(val),len(test))
    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                 f'train scaffolds = {train_scaffold_count:,} | '
                 f'val scaffolds = {val_scaffold_count:,} | '
                 f'test scaffolds = {test_scaffold_count:,}')
    return train,val,test

def read_smiles_property_file(args,
                              path, cols_to_read,
                              delimiter=',',
                              keep_header=True):
    reader = csv.reader(open(path, 'r'), delimiter=delimiter)
    data_full = np.array(list(reader))
    print(f'data_full.shape{data_full.shape}')
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        print(i,f'cols_to_read{cols_to_read}')
        data[i] = data_full[start_position:, col]
        if i >=1:

            data[i]=np.where(data[i]=='None',None,data[i] )
            data[i]=np.where(data[i]=='nan',None,data[i] )
            data[i]=np.where(data[i]=='',None, data[i])


    return data


def save_smiles_property_file(path, smiles, labels, delimiter=','):
    f = open(path, 'w')
    n_targets = labels.shape[1]
    for i in range(len(smiles)):
        f.writelines(smiles[i])
        for j in range(n_targets):
            f.writelines(delimiter + str(labels[i, j]))
        f.writelines('\n')
    f.close()


def read_split_scale_write( args,  data_path=None,
                      tmp_data_dir = None, cols_to_read=None):
    if args.data_path !=None:
        data_path=args.data_path
    if args.tmp_data_dir !=None:
        tmp_data_dir=args.tmp_data_dir

    data = read_smiles_property_file(args,args.data_path,
                                     args.cols_to_read,
                                     keep_header=False)

    smiles = data[0]
    if len(data) > 1:
        labels = np.array(data[1:], dtype='float')
        labels = labels.T
        print(f'labels looks like{labels}{labels.shape}')
        args.n_task=n_task=len(data)-1
    else:
        labels = None
        n_task=None
    try:
        os.stat(tmp_data_dir)
    except:
        os.mkdir(tmp_data_dir)

    cross_validation_split = KFold(n_splits=10, shuffle=True,random_state=args.seed)
    data = list(cross_validation_split.split(smiles, labels))
    i = 0
    sizes=(0.8,0.1,0.1)
    scalers=[]
    train_steps=[]
    args.class_weights=[]
    for split in data:
        if args.split_type == 'random':
            print('Cross validation with random split, fold number ' + str(i) + ' in progress...')
            train_val, test = split
            train, val =train_test_split(train_val, test_size=0.11111111111,random_state=args.seed)
        if args.split_type == 'scaffold_sort':
            scaf_seed=args.seed + i
            train,val,test=scaffold_split(smiles,sizes, scaf_seed,balanced=False,use_indices=True,big_small=2)
            print(f'using scaffold split ')
        if args.split_type == 'scaffold_balanced':
            scaf_seed=args.seed + i
            train,val,test=scaffold_split(smiles,sizes, scaf_seed,balanced=True,use_indices=True,big_small=2)
        if args.split_type == 'scaffold_random':
            scaf_seed=args.seed + i
            train,val,test=scaffold_split(smiles,sizes, scaf_seed,scaff_random=True,use_indices=True,big_small=2)
        X_train = smiles[train]
        train_steps.append(len(X_train)//args.batch_size)
        y_train = labels[train]

        X_val=smiles[val]
        y_val=labels[val]
        X_test=smiles[test]
        y_test=labels[test]
        args.train_size=len(X_train)
        args.val_size=len(X_val)
        args.test_size=len(X_test)

        if args.dataset_type == 'regression':
            if args.scale=="standardization":
                print('Fitting scaler(Z-score standardization)')
                scaler = StandardScaler().fit(y_train)
                y_train_scaled = scaler.transform(y_train)
                print(f'train data mean:{scaler.means}\nstd:{scaler.stds}\n')
            if args.scale=="normalization":
                print('Fitting scaler( Min-Max normalization )')
                scaler = minmaxScaler().fit(y_train)
                y_train_scaled = scaler.transform(y_train)
                print(f'train data min:{scaler.mins}\ntrain data max:{scaler.maxs}\n')
            if args.scale !='standardization' and args.scale!='normalization':
                raise ValueError("not implemented scaler,use one of [standardization, normalization]")
        else:
            scaler = None
        scalers.append(scaler)
        assert n_task != None
        save_smiles_property_file(f'{tmp_data_dir}{args.seed}{args.data_filename}_{i}_train',
                                  X_train, y_train.reshape(-1, n_task))
        if args.dataset_type=='classification':
            if args.class_balance:
                train_labels=y_train.reshape(-1, n_task).tolist()

                valid_targets = [[] for _ in range(args.n_task)]
                for ij in range(len(train_labels)):
                    for task_num in range(args.n_task):
                        if not math.isnan(train_labels[ij][task_num]):
                            valid_targets[task_num].append(train_labels[ij][task_num])
                train_class_sizes = []

                for task_targets in valid_targets:

                    assert set(np.unique(task_targets)) <= {0, 1}
                    try:
                        ones = np.count_nonzero(task_targets) / len(task_targets)
                    except ZeroDivisionError:
                        ones = float('nan')
                        print('Warning: class has no targets')
                    train_class_sizes.append([1 - ones, ones])
                class_batch_counts = torch.Tensor(train_class_sizes) * args.batch_size

                args.class_weights.append(1 / torch.Tensor(class_batch_counts))

        if args.dataset_type=='regression':
            save_smiles_property_file(f'{tmp_data_dir}{args.seed}{args.data_filename}_{i}_trainScaled',
                                  X_train, y_train_scaled.reshape(-1, n_task))

        save_smiles_property_file(f'{tmp_data_dir}{args.seed}{args.data_filename}_{i}_test',
                                  X_test, y_test.reshape(-1, n_task))
        save_smiles_property_file(f'{tmp_data_dir}{args.seed}{args.data_filename}_{i}_val',
                                  X_val, y_val.reshape(-1, n_task))
        i+=1
    print(f'train_steps:{train_steps}')
    return scalers, train_steps

def normalize_features( scaler: StandardScaler = None, replace_nan_token: int = 0, data: float=None) -> StandardScaler:
        if len(data) == 0 or data[0].features is None:
            return None
        if scaler is not None:
            scaler = scaler
        elif scaler is None:
            features = np.vstack([d.features for d in data])
            scaler = StandardScaler(replace_nan_token=replace_nan_token)
            scaler.fit(features)
        for d in data:
            d.set_features(scaler.transform(d.features.reshape(1, -1))[0])
        return scaler


class DGLDataset(Dataset):
    def __init__(self, dataf, training=True):
        print(f'Loading data...{dataf}')
        self.data_file = dataf
        with open(self.data_file) as f:
            self.data = [line.strip("\r\n") for line in f]
        print('Loading finished.')
        print('\tNum samples:', len(self.data))
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx].split(',')[0]
        lab_val=self.data[idx].split(',')

        label=[None if lab =='nan' or lab ==''or lab=='None' else lab for lab in lab_val][1:]
        label=[float(x) if x is not None else None for x in label]


        result = {
            'label': label,
            'sm': smiles
        }
        return result

def _unpack_field(examples, field):
    return [e[field] for e in examples]


class DGLCollator(object):
    def __init__(self, training):
        self.training = training

    def __call__(self, examples):



        mol_labels = _unpack_field(examples, 'label')
        mol_sms = _unpack_field(examples, 'sm')
        result_batch = {
            'sm': mol_sms,
            'labels': mol_labels,
        }


        return result_batch


def ensmable_train(args,logger,fold_i,train_dataloader,val_dataloader,test_dataloader,fold_path,scaler=None,epoch_steps=None):
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    if args.gpu is not None and args.gpuUSE:
        torch.cuda.set_device(args.gpu)
        print(f'USE GPU ID={args.gpu}')

    debug(pformat(vars(args)))

    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)
    sum_predicts=[]




    for model_idx in range(args.ensemble_size):

        save_dir = os.path.join(fold_path, f'model_{model_idx}')
        makedirs(save_dir)
        writer = SummaryWriter(log_dir=save_dir)

        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args)

        debug(model)
        debug(f'mdoel:{model_idx}>>>>Number of parameters = {param_count(model):,}')
        if args.gpuUSE:
            debug('Moving model to cuda')
            model = model.cuda()
        else:print('noGPU use')

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data,f'param.data is GPU{param.data.is_cuda}')

        save_checkpoint(os.path.join(save_dir, f'model{model_idx}.pt'), model, scaler, args)

        optimizer = build_optimizer(model, args)

        scheduler = build_lr_scheduler(optimizer, args, epoch_steps)


        print(f'args.minimize_score={args.minimize_score},args.metric={args.metric}')
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        hold_loss, hold_avgVal=[],[]
        for epoch in trange(1,args.epochs + 1):
            steps_eachEpoch, args.train_data_size,lastAvageloss, epoch_loss = train_batch(args,fold_i,model, train_dataloader, loss_func=loss_func,

                                                               optimizer=optimizer,
                                                               scheduler=scheduler,

                                                               logger=logger,
                                                               writer=writer)
            hold_loss.append(epoch_loss)
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            _, _, val_scores = evaluate_batch(args,
                model=model,
                data=val_dataloader,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                scaler=scaler,
                Foldth=args.Foldth,
                predsLog=None,
                logger=logger)

            avg_val_score = np.nanmean(val_scores)
            print(f'val_scores___{val_scores}')
            hold_avgVal.append(avg_val_score)

            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            writer.add_scalar(f'validation_{args.metric}_epoch', avg_val_score, epoch)
            writer.add_scalar(f'train_loss_epoch', lastAvageloss, epoch)
            if args.show_individual_scores:

                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    writer.add_scalar(f'validation_{task_name}_{args.metric}_epoch', val_score, epoch)

            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                print(f'debug args.minimize_score:{args.minimize_score} and {avg_val_score} < {best_score} ')
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, f'{args.data_filename}_model.pt'), model, scaler, args)

                info(f'Model {model_idx} the parametrs updated in model.pt as best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

        train_valdation_curve(args,fold_i,model_idx,cur_name='train-valdation curve')

        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        print(f'load model with args.cuda={args.cuda}')
        model = load_checkpoint(os.path.join(save_dir, f'{args.data_filename}_model.pt'), cuda=args.cuda, logger=logger)

        test_targets,test_preds,test_scores = evaluate_batch(args,
                                    model=model,
                                    data=test_dataloader,
                                    num_tasks=args.num_tasks,
                                    metric_func=metric_func,
                                    dataset_type=args.dataset_type,
                                    scaler=scaler,
                                    logger=logger,
                                    Foldth=args.Foldth,
                                    predsLog=args.save_dir)
        if len(test_preds) != 0:
            sum_predicts.append(np.stack(test_preds,axis=0))
            print(f'sum_predicts ={sum_predicts}')


        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test >>>  {args.metric} = {avg_test_score:.6f}')
        writer.add_scalar(f'test_{args.metric}_modelID', avg_test_score, model_idx)
        if args.show_individual_scores:

            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')
                writer.add_scalar(f'test_{task_name}_{args.metric}_ModelID', test_score, model_idx)
    sum_predict=np.zeros(sum_predicts[0].shape)
    print(len(sum_predicts))
    for mode_pred in sum_predicts:
        sum_predict =sum_predict + mode_pred

    avg_test_preds = (sum_predict / args.ensemble_size).tolist()
    if args.dataset_type == 'classification':
        all_classificationScores=evaluate_predictionsWithAllmetric(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func={'auc':roc_auc_score,'acc':acc_score,'precision':prec_score, 'recall':rec_score,'prec_auc':prec_rec_auc},
        dataset_type=args.dataset_type,
        logger=logger)
        print(f'use the metric {args.metric} for Hyperparameter Optimization')
        ensemble_scores=all_classificationScores[args.metric]
        all_metricsScore=all_classificationScores
    if args.dataset_type == 'regression':
        all_regressionScores=evaluate_predictionsWithAllmetric(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func={'rmse':rmse, 'mse':mean_squared_error,'mae':mean_absolute_error, 'r2':r2_score,'PC':Pearson_cor},
        dataset_type=args.dataset_type,
        logger=logger)
        print(f'use the metric {args.metric} for Hyperparameter Optimization')
        ensemble_scores=all_regressionScores[args.metric]
        all_metricsScore=all_regressionScores

    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    print(f'ensemble_scores={ensemble_scores} and test {args.metric} = {avg_ensemble_test_score:.6f}')
    writer.add_scalar(f'ensemble_test_{args.metric}_fold', avg_ensemble_test_score, fold_i)

    if args.show_individual_scores:
        for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
            info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
    return ensemble_scores,all_metricsScore

def train_batch(args,fold_i,model: nn.Module, data: DataLoader, loss_func: Callable, optimizer: Optimizer,
          scheduler: _LRScheduler,
          logger: logging.Logger = None,
          writer: SummaryWriter = None):
    debug = logger.debug if logger is not None else print
    loss_sum, iter_count,epoch_loss = 0, 0, 0
    for it, result_batch in enumerate(tqdm(data)):

        model.zero_grad()
        batch=result_batch['sm']
        label_batch=result_batch['labels']




        mask = torch.Tensor([[x is not None for x in batch_t] for batch_t in result_batch['labels']])
        targets = torch.Tensor([[0 if x is None else x for x in batch_t] for batch_t in result_batch['labels']])
        args.num_tasks = len(result_batch['labels'][0])
        if args.dataset_type=='classification':
            if args.class_balance:
                class_weights = []
                for task_num in range(args.n_task):
                    class_weights.append(args.class_weights[fold_i][task_num][targets[:, task_num].long()])
                class_weights = torch.stack(class_weights).t()

            else:
                class_weights = torch.ones(targets.shape)
        if next(model.parameters()).is_cuda and args.gpuUSE:

            mask, targets = mask.cuda(), targets.cuda()

        preds = model(batch)

        if args.dataset_type=='classification':
            loss = loss_func(preds, targets) * class_weights * mask
        else:
            loss = loss_func(preds, targets) *  mask

        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        epoch_loss +=loss.item()
        iter_count += targets.size(0)
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()

        if it % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / (iter_count*targets.size(0))
            loss_sum, iter_count = 0, 0
            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))

            if writer is not None:
                writer.add_scalar('train_loss_batch', loss_avg, it)
                writer.add_scalar('param_norm_batch', pnorm, it)
                writer.add_scalar('gradient_norm_batch', gnorm, it)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}_batch', lr, it)

    return it,it*targets.size(0),loss_avg,epoch_loss

def evaluate_batch(args:Namespace,
             model: nn.Module,
             data: DataLoader,
             num_tasks: int,
             metric_func: Callable,
             dataset_type: str,
             Foldth: int,
             scaler: StandardScaler = None,
             logger: logging.Logger = None,
             predsLog=None) -> List[float]:

    info = logger.info if logger is not None else print
    model.eval()
    preds = []
    targets=[]
    smiles=[]
    targets_sca=[]
    predsBack=[]
    if predsLog!=None:
        predsLog=Path(predsLog)/f'predsLogFoldth{Foldth}_{args.data_filename}'

    for it, result_batch in enumerate(tqdm(data)):

        model.zero_grad()
        batch_labels = result_batch['labels']
        batch_sm = result_batch['sm']
        with torch.no_grad():
            batch_preds = model(batch_sm)

        batch_preds = batch_preds.data.cpu().numpy()


        if scaler is not None:
            batch_preds_Sback = scaler.inverse_transform(batch_preds)
            batch_preds_Sback=batch_preds_Sback.tolist()
            predsBack.extend(batch_preds_Sback)

            batch_labels_sca = scaler.transform(batch_labels)
            targets_sca.extend(batch_labels_sca)


        preds.extend(batch_preds.tolist())
        targets.extend(batch_labels)
        smiles.extend(batch_sm)
    if predsLog!=None:
        with open(predsLog, 'w+') as pf:
            if scaler is not None:
                pf.write(f'smiles,labels,predictions,ORI_label,SB_pred,diff\n')
                for i, sm in enumerate(smiles):
                    lab_S=targets_sca[i]
                    pred=preds[i]
                    pred_SB=predsBack[i]
                    lab_noscle=targets[i]
                    diff=np.array(lab_noscle, dtype=np.float32) - np.array(pred_SB, dtype=np.float32)
                    pf.write(f'{sm},{lab_S},{pred},{lab_noscle},{pred_SB},{diff},<{i}>\n')
            else:
                pf.write(f'smiles,labels,predictions\n')
                for i, sm in enumerate(smiles):
                    pred_noScale=preds[i]
                    lab_noscle=targets[i]
                    pf.write(f'{sm},{pred_noScale},{lab_noscle}\n')

    if len(preds) == 0:
        return [float('nan')] * num_tasks
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]

    if args.dataset_type == 'regression':
        if scaler is None:

            targets=targets
        else:
            preds=predsBack



    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    results = []
    for i in range(num_tasks):

        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                info(f'Warning: {args.split_type}Found a task with targets all 0s or all 1s,try random split to aviod all 1s or 0s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                info('Warning: Found a task with predictions all 0s or all 1s')
            if nan:
                results.append(float('nan'))
                continue
        if len(valid_targets[i]) == 0:
            continue
        metric_func(valid_targets[i], valid_preds[i])
        results.append(metric_func(valid_targets[i], valid_preds[i]))

    return targets, preds, results

def write_metric_file(args,all_scores=None,all_metricsScores=[]):
    init_seed=args.seed
    if args.dataset_type=='regression':
        r2_, mae_, mse_, rmse_,Pearson_cor_, mape_=[],[],[],[],[],[]
        all_scores = np.array(all_scores)
        for fold_num, scores in enumerate(all_scores):
            r2_.append(all_metricsScores[fold_num]['r2'])
            rmse_.append(all_metricsScores[fold_num]['rmse'])
            mse_.append(all_metricsScores[fold_num]['mse'])
            mae_.append(all_metricsScores[fold_num]['mae'])
            Pearson_cor_.append(all_metricsScores[fold_num]['PC'])
            print(f'fold_num{fold_num} with seed {init_seed } ==> test {args.metric} = {np.nanmean(scores):.6f}')
            if args.show_individual_scores:
                for task_name, score in zip(args.task_names, scores):
                    print(f'Seed {init_seed} ==> test {task_name} {args.metric} = {score:.6f}')

        print(f'rmse {rmse_}\n mean:{np.nanmean(rmse_)}  std:{np.nanstd(rmse_)}\n')
        print(f'mae {mae_}\n mean:{np.nanmean(mae_)}  std:{np.nanstd(mae_)}\n')

        print(f'mse {mse_}\n mean:{np.nanmean(mse_)}  std:{np.nanstd(mse_)}\n')
        print(f'r2 {r2_}\n mean:{np.nanmean(r2_)}  std:{np.nanstd(r2_)}\n')
        print(f'Pearson_cor {Pearson_cor_}\n mean:{np.nanmean(Pearson_cor_)}  std:{np.nanstd(Pearson_cor_)}\n')
        if args.log_dir == None:
            print(args)
            with open(f'{args.data_filename}_log2jupyter','a') as wf:
                wf.write(f'RMSEs={np.nanmean(rmse_)},\n')
                wf.write(f'MSEs ={np.nanmean(mse_)},\n')
                wf.write(f'MAEs ={np.nanmean(mae_)},\n')
                wf.write(f'R2s={np.nanmean(r2_)},\n')
                wf.write(f'PCs={np.nanmean(Pearson_cor_)},\n')
                wf.write(f'stds_rmse={np.nanstd(rmse_)},\n')
                wf.write(f'stds_mse={np.nanstd(mse_)},\n')
                wf.write(f'stds_mae={np.nanstd(mae_)},\n')
                wf.write(f'stds_r2={np.nanstd(r2_)},\n')
                wf.write(f'stds_pc={np.nanstd(Pearson_cor_)},\n')
                wf.write(f'################################seed{args.seed}in {args}##########################\n')
    if args.dataset_type=='classification':
        auc_, acc_, prescision_, recall_, pre_auc_=[],[],[],[],[],[]
        all_scores = np.array(all_scores)
        for fold_num, scores in enumerate(all_scores):
            auc_.append(all_metricsScores[fold_num]['AUC'])
            prescision_.append(all_metricsScores[fold_num]['precision'])
            acc_.append(all_metricsScores[fold_num]['acc'])
            recall_.append(all_metricsScores[fold_num]['recall'])
            pre_auc_.append(all_metricsScores[fold_num]['prec_auc'])
            print(f'fold_num{fold_num} with seed {init_seed } ==> test {args.metric} = {np.nanmean(scores):.6f}')
            if args.show_individual_scores:
                for task_name, score in zip(args.task_names, scores):
                    print(f'Seed {init_seed} ==> test {task_name} {args.metric} = {score:.6f}')
        print(f'precision {prescision_}\n mean:{np.nanmean(prescision_)}  std:{np.nanstd(prescision_)}\n')
        print(f'ACC {acc_}\n mean:{np.nanmean(acc_)}  std:{np.nanstd(acc_)}\n')
        print(f'prec_auc {pre_auc_}\n mean:{np.nanmean(pre_auc_)}  std:{np.nanstd(pre_auc_)}\n')
        print(f'AUC {auc_}\n mean:{np.nanmean(auc_)}  std:{np.nanstd(auc_)}\n')
        print(f'recall {recall_}\n mean:{np.nanmean(recall_)}  std:{np.nanstd(recall_)}\n')
        if args.log_dir == None:
            print(args)
            with open(f'{args.data_filename}_log2jupyter','a') as wf:
                wf.write(f'AUCs={np.nanmean(auc_)},\n')
                wf.write(f'Precisions ={np.nanmean(prescision_)},\n')
                wf.write(f'ACCs ={np.nanmean(acc_)},\n')
                wf.write(f'Recalls={np.nanmean(recall_)},\n')
                wf.write(f'Pre_aucs={np.nanmean(pre_auc_)},\n')

                wf.write(f'stds_AUC={np.nanstd(auc_)},\n')
                wf.write(f'stds_Preci={np.nanstd(prescision_)},\n')
                wf.write(f'stds_Recal={np.nanstd(recall_)},\n')
                wf.write(f'stds_acc={np.nanstd(acc_)},\n')
                wf.write(f'stds_Pre_auc={np.nanstd(pre_auc_)},\n')
def evaluate_predictionsWithAllmetric(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         dataset_type: str,
                         metric_func: dict,
                         logger: logging.Logger = None) -> List[float]:
    info = logger.info if logger is not None else print
    if len(preds) == 0:
        return [float('nan')] * num_tasks


    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]

    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])


    results = {}
    task_results=[[] for _ in range(len(metric_func)) ]
    li=0
    keep_nan_metrics=False
    for mf in metric_func:

        for i in range(num_tasks):

            if dataset_type == 'classification':
                nan = False
                if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                    nan = True
                    print(f'Warning: Found a task with targets all 0s or all 1s,try random split to aviod all 1s or 0s')
                if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                    nan = True
                    print(f'Warning: Found a task with predictions all 0s or all 1s in task col{i}')
                if nan:
                    if keep_nan_metrics:
                        if mf == 'auc':
                            results[mf]=0.5
                        elif mf in ['prc-auc', 'accuracy']:
                            results[mf]=0

                    else:
                        results[mf]=float('nan')
                    continue
            if len(valid_targets[i]) == 0:
                continue
            task_results[li].append(metric_func[mf](valid_targets[i], valid_preds[i]))
            if mf=='precision' or mf=='recall':
                print(f'mf={mf}\nvalid_preds{i}  {metric_func[mf](valid_targets[i], valid_preds[i])}')
        results[mf]=task_results[li]
        li+=1
    return results

def cv(args: Namespace, logger: Logger = None)-> Tuple[float, float]:
    """Beware running 2 processes on the same machine, even if you set random seeds and set num_workers=0. In my experience running one process on it’s own is deterministic, but running 2 processes side by side is not. The consensus here is that static variables in dynamically linked libraries are to blame. If you need to compare, run on two different machines.
(by 2 processes I mean two training scripts for example) """
    def worker_init_fn():


        np.random.seed(args.seed)

    def seed_torch(seed=args.seed):
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

    seed_torch(args.seed)


    args.ffn_hidden_size =300

    args.minimize_score = args.metric in ['rmse', 'mae','mse']
    args.data_filename=os.path.basename(args.data_path)+f'_seed{args.seed}'
    info = logger.info if logger is not None else print

    init_seed = args.seed
    task_names = get_task_names(args.data_path)
    args.task_names=task_names
    args.num_tasks=len(task_names)

    print(f'task_name are:{task_names}\nnumber of tasks are:{args.num_tasks}')
    all_scores = []
    all_metricsScores=[]
    direct = args.tmp_data_dir
    fold = args.num_folds
    scalers,models_stepsEachepoch=read_split_scale_write(args,args.data_path,args.tmp_data_dir,cols_to_read=args.cols_to_read)
    for f_i in range(fold):
        args.Foldth=f_i
        fold_path=os.path.join(args.save_dir, f'fold_{f_i}')
        makedirs(fold_path)
        DGLval = DGLDataset(f'{direct}{args.seed}{args.data_filename}_{f_i}_val', training=False)
        DGLtest = DGLDataset(f'{direct}{args.seed}{args.data_filename}_{f_i}_test', training=False)
        if args.dataset_type == 'classification':
            DGLtrain = DGLDataset(f'{direct}{args.seed}{args.data_filename}_{f_i}_train', training=True)
        else:
            DGLtrain = DGLDataset(f'{direct}{args.seed}{args.data_filename}_{f_i}_trainScaled', training=True)

        train_dataloader = DataLoader(DGLtrain, batch_size=args.batch_size,
                                      shuffle=True, num_workers=0,
                                      collate_fn=DGLCollator(training=True),
                                      drop_last=False, worker_init_fn=worker_init_fn)
        val_dataloader = DataLoader(DGLval, batch_size=args.batch_size,
                                    shuffle=True, num_workers=0,
                                    collate_fn=DGLCollator(training=False),
                                    drop_last=False,worker_init_fn=worker_init_fn)
        test_dataloader = DataLoader(DGLtest, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0,
                                     collate_fn=DGLCollator(training=False),
                                     drop_last=False,
                                     worker_init_fn=worker_init_fn)
        model_scores,all_metricsScore = ensmable_train(args, logger,f_i,train_dataloader,val_dataloader,test_dataloader,fold_path,scaler=scalers[f_i],epoch_steps=models_stepsEachepoch[f_i])
        all_scores.append(model_scores)
        all_metricsScores.append(all_metricsScore)
    all_scores = np.array(all_scores)
    print('all_scores',all_scores)
    print('all_metricsScores',all_metricsScores)

    write_metric_file(args,all_scores,all_metricsScores)

    with open(f'{args.data_filename}_AllfoldsMetrics.txt','a+') as wf:
        wf.write(str(all_metricsScores))

    avg_scores = np.nanmean(all_scores, axis=1)


    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    print(f'all_scores={all_scores}\navg_scores={avg_scores}\nmean_score={mean_score}')




    info(f'Overall test from fold{f_i} :{args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    if args.show_individual_scores:
        if len((args.task_names))<=1:
            task_num=1
            info(f'show_individual_scores {args.task_names} {args.metric} = {all_scores} +/- 0 as task_num{task_num}')
        else:
            for task_num, task_name in enumerate(args.task_names):
                info(f'Overall test {task_name} {args.metric} = {np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score

