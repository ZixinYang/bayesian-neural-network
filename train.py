import torch
import torch.nn as nn
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from nn import NN
from model import model, guide
from data import data_download
import numpy as np

optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())
num_iterations = 5
loss = 0

train_data, test_data = data_download()

for j in range(num_iterations):
    loss = 0
    for batch_id, data in enumerate(train_data):
        loss += svi.step(data[0].view(-1,28*28), data[1])
    normalizer_train = len(train_data.dataset)
    total_epoch_loss_train = loss / normalizer_train
    
    print("Epoch ", j, " Loss ", total_epoch_loss_train)

pyro.get_param_store().save('saved_params.save')