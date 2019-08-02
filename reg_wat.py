from parsing import parse_train_args
from cv import cv
from utils import create_logger




"""
source ~/conda_py.sh
source activate torch
"""

if __name__ == '__main__':
    args = parse_train_args()

    args.num_folds=10
    args.epochs=50
    args.ensemble_size=1
    args.batch_size=128

    args.activation='ReLU'
    args.depth=4
    args.dropout=0.25
    args.ffn_num_layers=2
    args.hidden_size=384
    args.sumstyle=True
    args.seed=3032
    args.gpuUSE=True
    args.gpu=3

    args.data_path,args.cols_to_read ='data_RE2/water_solubilityOCD.csv',[x for x in range(2)]




    args.save_dir='save_test'

    args.tmp_data_dir='./data_RE2/tmp/'
    args.scale='normalization'




    args.split_type='random'




    args.diff_depth_weights=True
    args.layers_per_message=1


    args.attention=True
    args.message_attention=False
    args.global_attention=False
    args.message_attention_heads=1
    args.log_dir=None

    print(args)
    logger = create_logger(name='train_crossValidate', save_dir=args.save_dir, quiet=args.quiet)



    cv(args, logger)









