import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import scipy.misc
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import torchvision
import numpy as np
import torch.utils.data as data_utils
import torch.nn.functional as F
from data_utils import *


# lr_scheduler() manages the learning rate according to given condition

def lr_scheduler(optimizer, init_lr, epoch):
    
    
    for param_group in optimizer.param_groups:
        
        if epoch == 150 or epoch == 225:
            param_group['lr']=param_group['lr']/10
            
        if epoch == 0:
            param_group['lr']=init_lr
        
        print('Current learning rate is {}'.format(param_group['lr']))


    return optimizer






def train_model(cnn,optimizer_s,lrate,num_epochs,train_loader,test_loader,dataset_train_len,dataset_test_len, plotsFileName, csvFileName):

    epochs= []
    train_acc=[]
    test_acc=[]
    train_loss=[]
    test_loss = []
    train_error=[]
    test_error =[]

    for epoch in range(num_epochs):
        cnn.train()
        epochs.append(epoch)
        optimizer  = lr_scheduler(optimizer_s, lrate, epoch)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('*' * 70)
        running_loss = 0.0
        running_corrects = 0.0
        train_batch_ctr = 0.0
        
        for i, (image, label) in enumerate(train_loader):
   
                image,label = Variable(image.cuda(),requires_grad=True),Variable(label.cuda(),requires_grad=False)
               
                optimizer.zero_grad()

              
                outputs = cnn(image)
               
                _, preds = torch.max(outputs.data, 1)
                loss = F.nll_loss(outputs, label)
                loss.backward()
                optimizer.step()
                train_batch_ctr = train_batch_ctr + 1


               
                running_loss += loss.data[0]

                running_corrects += torch.sum(preds == label.data)
                
                epoch_acc = running_corrects / (dataset_train_len)

        print ('Train corrects: {} Train samples: {} Train accuracy: {}' .format( running_corrects, (dataset_train_len),epoch_acc))
        train_acc.append(epoch_acc)
        train_loss.append(running_loss / train_batch_ctr)
        train_error.append(((dataset_train_len)-running_corrects) / (dataset_train_len))


        cnn.eval()  
        test_running_corrects = 0.0
        test_batch_ctr = 0.0
        test_running_loss = 0.0
        test_total = 0.0
       
        for image, label in test_loader:
           
            image, label = Variable(image.cuda(),volatile=True), Variable(label.cuda())
            
            test_outputs = cnn(image)
            _, predicted_test = torch.max(test_outputs.data, 1)
            
            loss = F.nll_loss(test_outputs, label)
            test_running_loss += loss.data[0]
            test_batch_ctr = test_batch_ctr+1
            
            test_running_corrects += torch.sum(predicted_test == label.data)
            test_epoch_acc = test_running_corrects / (dataset_test_len)

        
        test_acc.append(test_epoch_acc)
        test_loss.append(test_running_loss / test_batch_ctr)
        test_error.append(((dataset_test_len)-test_running_corrects) / (dataset_test_len))
       
        print('Test corrects: {} Test samples: {} Test accuracy {}' .format(test_running_corrects,(dataset_test_len),test_epoch_acc))
        
        print('Train loss: {} Test loss: {}' .format(train_loss[epoch],test_loss[epoch]))
        
        print('Train error: {} Test error {}' .format(train_error[epoch],test_error[epoch]))
        
        print('*' * 70)
       
        
        plots(epochs, train_acc, test_acc, train_loss, test_loss,train_error,test_error,plotsFileName)
        write_csv(csvFileName, train_acc,test_acc,train_loss,test_loss,train_error,test_error,epoch)
        
        
        '''
         plots() and write_csv() are defined in data_utils.py. plots() updates the training plots with each epoch and 
         write_csv() updates training log with each epoch.
        '''
        
     



