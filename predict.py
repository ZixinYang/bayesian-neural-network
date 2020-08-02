import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from nn import NN
from model import model, guide
from data import data_download

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

num_samples = 100
def give_uncertainities(x):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [F.log_softmax(model(x.view(-1,28*28)).data, 1).detach().numpy() for model in sampled_models]
    return np.asarray(yhats)

def test_batch(images, labels):
    y = give_uncertainities(images)
    predicted_for_images = 0
    correct_predictions=0
    for i in range(len(labels)):
        all_digits_prob = []
        highted_something = False
        for j in range(len(classes)):    
            highlight=False 
            histo = []
            histo_exp = []
            for z in range(y.shape[0]):
                histo.append(y[z][i][j])
                histo_exp.append(np.exp(y[z][i][j]))    
            prob = np.percentile(histo_exp, 50)
            if(prob>0.2):
                highlight = True
            all_digits_prob.append(prob)
            if(highlight):       
                highted_something = True            
        predicted = np.argmax(all_digits_prob)
        if(highted_something):
            predicted_for_images+=1
            if(labels[i].item()==predicted):
                correct_predictions +=1.0        
    return len(labels), correct_predictions, predicted_for_images

pyro.get_param_store().load('saved_params.save')

train_data, test_data = data_download()

print('Prediction when network can refuse')
correct = 0
total = 0
total_predicted_for = 0
for j, data in enumerate(test_data):
    images, labels = data
    
    total_minibatch, correct_minibatch, predictions_minibatch = test_batch(images, labels)
    total += total_minibatch
    correct += correct_minibatch
    total_predicted_for += predictions_minibatch

print("Total images: ", total)
print("Skipped: ", total-total_predicted_for)
print("Accuracy when made predictions: %d %%" % (100 * correct / total_predicted_for))