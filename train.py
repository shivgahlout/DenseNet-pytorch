import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import math
import torch.utils.data as data_utils
import torch.nn.functional as F

from train_utils import*
from densenet import*



''' Main file. Run it to train the DenseNet. This file use train_utils.py and densenet.py '''




''' data augmentation and preprocessing '''

data_transforms = {
    'train': transforms.Compose([
     transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247032, 0.243485, 0.261587)),
    ]),
    'val': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247032, 0.243485, 0.261587)),
    ]),
}


''' Loading CIFAR10 dataset. If dataset is not available it will be downloaded.'''

dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=1)

dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=1)
dataset_test_len=len(dataset_test)
dataset_train_len=len(dataset_train)



''' 
    Defining Densenet using densenet() which is defined in densenet.py
    
    densenet() takes following arguments:
    
    growth_rate ---> growth rate of DenseNet; default 12
    depth ---> depth of DenseNet i.e. 40, 100 etc. ; default 100
    blocks ---> no of blocks in DenseNet; default 3
    bNeck ---> Boolean, if bottleneck is used or not; default TRUE
    compression ---> compression factor; default .5
    dropout ---> probability of dropout; default 0.0
'''

cnn = densenet(growth_rate=12,depth=100, blocks=3, bNeck=True, compression=0.5,dropout=0.0)


cnn = cnn.cuda()
print(cnn)

parameter=0

for param in cnn.parameters():
    parameter+=param.data.nelement()


print('Total trainable parameters are {}'.format(parameter))


lrate=.1   #initial learnig rate
optimizer_s = optim.SGD(cnn.parameters(), lr=lrate, momentum=0.9, weight_decay=1e-4)
num_epochs = 300
plotsFileName='./plots/densenet100BC_C10+' #Filename to save plots. Three plots are updated with each epoch; Accuracy, Loss and Error Rate
csvFileName='./stats/densenet100BC_C10+_log.csv' #Filename to save training log. Updated with each epoch, contains Accuracy, Loss and Error Rate



train_model(cnn, optimizer_s,lrate,num_epochs,train_loader,test_loader,dataset_train_len, dataset_test_len,plotsFileName,csvFileName)


'''
    train_model() is defined in train_utils.py and is used for training the model.
    
    train_model() takes following arguments:
    
    cnn ---> model to be trained
    optimizer_s ---> optimizer to be used
    lrate ---> initial learnig rate
    num_epochs ---> total no of epochs
    train_loader ---> train data
    test_loader ---> test data
    dataset_train_len ---> total train samples
    dataset_test_len ---> total test samples
    plotsFileName ---> file to save plots
    csvFileName ---> file to save training log
'''
    
    
    
    
    
    
    
    









