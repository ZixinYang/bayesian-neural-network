import numpy as np
import torch
import torch.nn as nn
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from nn import NN
from model import model, guide
from data import data_download

net = NN(28*28, 1024, 10)

pyro.get_param_store().load('saved_params.save')

train_data, test_data = data_download()

num_samples = 10
def predict(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.numpy(), axis=1)

print('Prediction when network is forced to predict')
correct = 0
total = 0
for j, data in enumerate(test_data):
    images, labels = data
    predicted = predict(images.view(-1,28*28))
    total += labels.size(0)
    correct += (predicted == np.array(labels)).sum().item()
print("accuracy: %d %%" % (100 * correct / total))