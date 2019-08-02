
# A Self-Attention Based Message Passing Neural Network for Identifying Structure-Property Relationships

## Introduction
This is a PyTorch implementation of the research: [A Self-Attention Based Message Passing Neural Network for Identifying Structure-Property Relationships](https://github.com/tbwxmu/SAMPN/SAMPN_git.png)
![Graph abstract](https://github.com/tbwxmu/SAMPN/SAMPN_git.png) 

## Environment
```
Python 3.6.5
Pytorch 1.0 
RDkit 2018.03.4 
Autograd 1.2 
Numpy 1.14.2 
Pandas 0.23.4 
tqdm 3.7.1
```
## Data File
Data file format: </br>
&nbsp;&nbsp;&nbsp;&nbsp;Datafile can be CSV and text as showd in the data_RE2. </br>


### train 
`python reg_wat.py #replace the data_path and cols_to_read as your want` </br>
The trained model weights will be also stored in `save_test`. </br>
### prediction and visualization
`python viz_wat.py #replace the data_path and checkpoint_path as your want` </br>
The results of prediction will be also stored in `save_test` and visualization file will be save in png_*. </br>
### repeat this work
bash go_repeat.sh

## Acknowledgement
We thank the previous work by the swansonk14 team. The code in this repository is inspired on [ChemProp](https://github.com/swansonk14/chemprop)
