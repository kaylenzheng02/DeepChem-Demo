import truststore
truststore.inject_into_ssl()

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import deepchem as dc
# imported directly - in this dc build, dc.models only exposes
# multitalkregressor when tensorflow installed
from deepchem.models.fcnet import MultitaskRegressor

#fix random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# load & featurize the esol dataset
print("loading and featurizing esol dataset...")
tasks, datasets, transformers = dc.molnet.load_delaney(
    featurizer="ECFP",
    splitter="scaffold"
)
train_dataset, valid_dataset, test_dataset = datasets

# initialize pyttorch mlp model my little pony 
model = MultitaskRegressor(
    n_tasks=len(tasks),
    n_features=1024,
    layer_sizes=[1000],
    dropouts=0.2
    learning_rate=0.001
)

# 3 wow train model 
print("training model...")
model.fit(train_dataset, nb_epoch=30)

# 4 eval neural network w metrics RMSE, MAE, R2
metrics = [
    dc.metrics.Metric(dc.metrics.rms_score, name="RMSE"),
    dc.metrics.Metric(dc.metrics.mae_score, name="MAE"),
    dc.metrics.Metric(dc.metrics.r2, name="R2"),
]
