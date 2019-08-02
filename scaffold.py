from collections import defaultdict
from copy import deepcopy
import logging
import random
from typing import Dict, List, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np

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

def scaffold_to_smiles(all_smiles: List[str], use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    scaffolds = defaultdict(set)  #http://kodango.com/understand-defaultdict-in-python
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
                   big_small: int=2
                   ):
    assert sum(sizes) == 1
    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    print(f'the splited size train val tset{train_size},{val_size}，{test_size}')
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0
    scaffold_to_indices = scaffold_to_smiles(data, use_indices=use_indices)
    if balanced:
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for scaff,index_set in scaffold_to_indices.items():
            if len(index_set) > val_size / big_small or len(index_set) > test_size / big_small:#考虑极端情况（如只有small），这里只为了打乱顺序，test 划分是分开的
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
        print(f'big_index_sets  {len(big_index_sets)} + small_index_sets  {len(small_index_sets)}')
        print(f'index_sets {len(index_sets)}')
    else:  # Sort from largest to smallest scaffold sets
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
    #print(test)
    print(len(index_sets))
    print('train val test',len(train),len(val),len(test))
    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                 f'train scaffolds = {train_scaffold_count:,} | '
                 f'val scaffolds = {val_scaffold_count:,} | '
                 f'test scaffolds = {test_scaffold_count:,}')
    return train,val,test