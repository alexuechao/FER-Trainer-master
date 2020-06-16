"""
plot confusion_matrix of PublicTest and PrivateTest
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
#from utils import dataloader
from utils.dataloader import DataLoader
from torch.autograd import Variable
import torchvision
import transforms as transforms
from sklearn.metrics import confusion_matrix
from models import *
from models.resnet_cut import *
from models.peleenet import *

parser = argparse.ArgumentParser(description='plot_rgb_confusion_matrix')
parser.add_argument('--model_path', help='input model path', type=str)
parser.add_argument('--input_shape', help='data type', default=128, type=int)
parser.add_argument('--split', type=str, default='PrivateTest', help='split')
opt = parser.parse_args()

input_shape = opt.input_shape
transform_test = transforms.Compose([
    transforms.CenterCrop(input_shape),
    transforms.ToTensor(),
    #normalize,
])
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Model
device_ids=[0]
#net = ResNet18_cut()
net = Peleenet()
net = torch.nn.DataParallel(net, device_ids=device_ids)
checkpoint = torch.load(opt.model_path)
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()
PrivateTestset = DataLoader(split = opt.split, transform=transform_test)
Testloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=16, shuffle=False)
correct = 0
total = 0
all_target = []
for batch_idx, (inputs, targets) in enumerate(Testloader):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        inputs, targets = inputs.cuda(device=device_ids[0]), targets.cuda(device=device_ids[0])
    inputs, targets = Variable(inputs), Variable(targets)
    outputs = net(inputs)
    score = F.softmax(outputs)
    #print(score)
    _, predicted = torch.max(outputs, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()
    # inputs, targets = Variable(inputs), Variable(targets)
    # outputs = net(inputs)
    # _, predicted = torch.max(outputs.data, 1)

    # total += targets.size(0)
    # correct += predicted.eq(targets.data).cpu().sum()
    if batch_idx == 0:
        all_predicted = predicted
        all_targets = targets
    else:
        all_predicted = torch.cat((all_predicted, predicted),0)
        all_targets = torch.cat((all_targets, targets),0)

acc = 100. * float(correct) / total
print("accuracy: %0.3f" % acc)

# Compute confusion matrix
matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                      title= opt.split+' Confusion Matrix (Accuracy: %0.3f%%)' %acc)
plt.savefig(os.path.join('./output_results/', opt.split + '_cm.png'))
plt.close()