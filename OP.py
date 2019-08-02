from argparse import ArgumentParser, Namespace
from parsing import parse_train_args
from copy import deepcopy
import json
from typing import Dict, Union
import os
import hyperopt
from hyperopt import fmin, hp, tpe
import numpy as np

from DGLmodels import build_model
from nn_utils import param_count
from parsing import add_train_args, modify_train_args
from cv import cv
from utils import create_logger, makedirs

SPACE = {
    'hidden_size': hp.quniform('hidden_size', low=100, high=300, q=50),
    'batch_size': hp.quniform('batch_size', low=32, high=128, q=32),

    'depth': hp.quniform('depth', low=2, high=6, q=1),

    'dropout': hp.quniform('dropout', low=0.0, high=0.4, q=0.05),
    'ffn_num_layers': hp.quniform('ffn_num_layers', low=1, high=3, q=1)
}
INT_KEYS = [
            'hidden_size',
            'batch_size',

            'depth',

            'dropout',
            'ffn_num_layers'
           ]


def grid_search(args: Namespace):

    logger = create_logger(name='hyperparameter_optimization', save_dir=args.log_dir, quiet=True)
    train_logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)

    results = []

    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:

        for key in INT_KEYS:
            print(f'hyperparams[key]={hyperparams[key]}')
            if type(hyperparams[key]) is not str:
                hyperparams[key] = int(hyperparams[key])
            else:
                args.activation=hyperparams[key]

        hyper_args = deepcopy(args)
        if args.save_dir is not None:
            folder_name = '_'.join(f'{key}_{value}' for key, value in hyperparams.items())
            hyper_args.save_dir = os.path.join(hyper_args.save_dir, folder_name)
        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)
        print(f'the used args====>>{args}')

        logger.info(hyperparams)

        mean_score, std_score = cv(hyper_args, train_logger)

        temp_model = build_model(hyper_args)
        num_params = param_count(temp_model)
        logger.info(f'num params: {num_params:,}')
        logger.info(f'{mean_score} +/- {std_score} {hyper_args.metric}')
        results.append({'mean_score': mean_score,
                        'std_score': std_score,
                        'hyperparams': hyperparams,
                        'num_params': num_params})

        if np.isnan(mean_score):
            if hyper_args.dataset_type == 'classification':
                mean_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')
        return (1 if hyper_args.minimize_score else -1) * mean_score

    best_=fmin(objective, SPACE, algo=hyperopt.rand.suggest, max_evals=args.num_iters)


    results = [result for result in results if not np.isnan(result['mean_score'])]
    best_result = min(results, key=lambda result: (1 if args.minimize_score else -1) * result['mean_score'])
    print(f'best_result={best_result}\n'
          ,f'best_={best_}')
    logger.info(f'best in the search space with seed={args.seed}')
    logger.info(best_result['hyperparams'])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(f'{best_result["mean_score"]} +/- {best_result["std_score"]} {args.metric}')

    makedirs(args.config_save_path, isfile=True)
    with open(args.config_save_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)



if __name__ == '__main__':
    parser = ArgumentParser()
    add_train_args(parser)
    args = parse_train_args()

    args.layers_per_message=1
    args.num_folds=1
    args.epochs=3
    args.ensemble_size=1

    args.data_path='data_RE2/JAK2_sci_ExCAPEDB.csv'

    args.save_dir='save_op'
    args.cols_to_read=[0,1]

    args.gpuUSE=False
    args.diff_depth_weights=False
    args.gpu=3
    args.dataset_type='regression'

    args.tmp_data_dir='./data_RE2/tmp/'

    args.scale='standardization'
    args.metric='r2'


    args.no_features_scaling=True

    args.attention=False
    args.message_attention=False
    args.global_attention=False
    args.message_attention_heads=1

    args.num_iters=10
    args.config_save_path='save_OP/bestOP.json'
    args.log_dir='save_OP'

    modify_train_args(args)

    args.minimize_score = args.metric in ['rmse', 'mae','mse']
    grid_search(args)





