# DenseNet-pytorch
Implementation of DenseNet Using Pytorch


##### This repository is developed using pytorch 0.2.0.

The [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) has bagged best paper award in CVPR 2017. Official implementation and oral presentation of this paper can be found [here](https://github.com/liuzhuang13/DenseNet) and [here](https://www.youtube.com/watch?v=-W6y8xnd--U) respectively. 


#### Results using DenseNet

We tried two versions of DenseNet, densenet-40 for C10 and densenet-100-BC for C10+. The official results for these settings are shown in figure below:

![](/images/table.png)

This implementation is achieving same results (error rate of 4.51) for C10+ using densenet-100-BC while for C10 using densenet-40 we are getting error rate of 7.91.

The error rate graph for C10+ using densenet-100-BC is: 

<img src = "/plots/densenet100BC_C10+_error.png"  width ="600">


The error rate graph for C10 using densenet-40 is:

<img src = "/plots/densenet_40_C10_error.png"  width ="600">

#### How to use this repository

You have to run train.py
###### python train.py

Any modifcation in densenet architecture can be made in train.py itself. More information about implementation can be found in files themselves. As a overview, train.py trains the model, densenet.py defines the densenet, data_utils.py and train_utils.py are supporting files. Three graphs showing accuracy, error rate and loss are updated with each epoch. Training log is also updated with each epoch in a .csv file. 

In case of any issue feel free to contact me. 
