import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
from scipy.misc import imsave, imread


def plots(epochs, train_acc, test_acc, train_loss, test_loss, train_error, test_error,filename):
    plt.style.use('bmh')

    fig=plt.figure(figsize=(8,6))
    plt.plot(epochs,train_acc,  'r', epochs,test_acc, 'g')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'test_acc'], loc='upper left')
    fig.savefig(filename + '_accuracy.png')

    fig=plt.figure(figsize=(8,6))
    plt.plot(epochs,train_loss,  'r', epochs,test_loss, 'g')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'test_loss'], loc='upper left')
    fig.savefig(filename + '_loss.png')
    
    fig=plt.figure(figsize=(8,6))
    plt.plot(epochs,train_error,  'r', epochs,test_error, 'g')
    plt.title('model error rate')
    plt.ylabel('error rate')
    plt.xlabel('epoch')
    plt.legend(['train_error', 'test_error'], loc='upper left')
    fig.savefig(filename + '_error.png')

    plt.close('all')



def write_csv(filename, train_acc,test_acc,train_loss,test_loss,train_error,test_error,epoch):
    if epoch==0:
        
        with open(filename, 'w') as f:
            f.write('train_acc,test_acc,train_loss, test_loss, train_error, test_error\n') 
            f.write('{0},{1},{2},{3},{4},{5}\n'.format(train_acc[-1],\
                                         test_acc[-1],\
                                          train_loss[-1],\
                                           test_loss[-1],\
                                           train_error[-1],\
                                           test_error[-1]))
            
    else:
        with open(filename, 'a') as f:
            f.write('{0},{1},{2},{3},{4},{5}\n'.format(train_acc[-1],\
                                         test_acc[-1],\
                                          train_loss[-1],\
                                           test_loss[-1],\
                                           train_error[-1],\
                                           test_error[-1]))
  


