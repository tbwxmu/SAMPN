#   set PYTHONPATH=%PYTHONPATH%;D:\workDir\deepchem
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import time
import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc



delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney(
    featurizer='Weave', split='random')

train_dataset, valid_dataset, test_dataset = delaney_datasets

metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

n_atom_feat = 75
n_pair_feat = 14

batch_size = 64

model = dc.models.MPNNModel(
    len(delaney_tasks),
    n_atom_feat=n_atom_feat,
    n_pair_feat=n_pair_feat,
    T=3,
    M=5,
    batch_size=batch_size,
    learning_rate=0.0001,
    use_queue=False,
    mode="regression")

start_time = time.time()
model.fit(train_dataset, nb_epoch=50, checkpoint_interval=100)
print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)
print(f" takes time:{round(time.time() - start_time,3)/60} minutes  ")
print("Train scores")
print(train_scores)
print("Validation scores")
print(valid_scores)
print("test scores")
print(test_scores)
