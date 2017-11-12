import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math



#implements DenseNet



#addLayers() adds layers in a block using suBlock()

def addLayers(inMap, growth_rate,bNeck, layers,dropout):
    


        layer=[]
        
       
            
        for l in range(layers):

            inMap1 = inMap+growth_rate*l
            layer.append(suBlock(inMap1, growth_rate,bNeck,dropout))


        return nn.Sequential(*layer)

        
        

class suBlock(nn.Module):
    def __init__(self, inMap, growth_rate,bNeck,dropout):
        super(suBlock, self).__init__()
        
        self.bNeck=bNeck
        self.dropout=dropout
        
        if self.bNeck:
            self.batch1=nn.BatchNorm2d(inMap)
            self.conv1=nn.Conv2d(inMap, 4*growth_rate, kernel_size=1,bias=False)
            self.batch=nn.BatchNorm2d(4*growth_rate)
            self.conv=nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1,bias=False)
            
            
            
        else:
            self.batch=nn.BatchNorm2d(inMap)
            self.conv=nn.Conv2d(inMap, growth_rate, kernel_size=3,padding=1,bias=False)
            
        
        
        self.drop=nn.Dropout2d(p=self.dropout)
        
        

    def forward(self, x):
        
        if self.bNeck:
        
            out=self.conv1(F.relu(self.batch1(x)))
            if self.dropout:
                out =self.drop(out)
            out=self.conv(F.relu(self.batch(out)))
            if self.dropout:
                out =self.drop(out)
            
        else:
            out=self.conv(F.relu(self.batch(x)))
            if self.dropout:
                out =self.drop(out)
            
        
        
        return torch.cat((out,x),1)
                 

        
        
#transLayer() adds transition layer

class transLayer(nn.Module):
    def __init__(self, inMap, outMap,dropout):
        
        
        super(transLayer, self).__init__()

        self.dropout=dropout
        self.batch=nn.BatchNorm2d(inMap)
        self.conv=nn.Conv2d(inMap, outMap, kernel_size=1,bias=False)
        self.pool=nn.AvgPool2d(kernel_size=2, stride=2)
        self.drop=nn.Dropout2d(p=self.dropout)

    def forward(self, x):
        
        x=self.conv(F.relu(self.batch(x)))
        if self.dropout:
                x =self.drop(x)
        x=self.pool(x)
        
        
        return x
    
    
    
class densenet(nn.Module):
    def __init__(self, growth_rate=12,depth=100, blocks=3, bNeck=True,compression=.5, dropout=0.0):
        super(densenet, self).__init__()
        
       
        
        if bNeck:
            inMap=2*growth_rate
            layers=(depth-blocks-1)//(2*blocks)
            
        else:
            inMap=16
            layers=(depth-blocks-1)//blocks
            
        self.conv1 = nn.Conv2d(3, inMap, kernel_size=3, padding=1,bias=False)
        
        self.model=addLayers(inMap,growth_rate,bNeck,layers,dropout)
        inMap+=layers*growth_rate
        if compression:
            outMap=int(math.floor(inMap*compression))
            
        else:
            outMap=inMap
            
        
        self.tr=transLayer(inMap, outMap,dropout)
        
        
        inMap=outMap
        self.model1=addLayers(inMap,growth_rate,bNeck,layers,dropout)
        inMap+=layers*growth_rate
        if compression:
            outMap=int(math.floor(inMap*compression))
            
        else:
            outMap=inMap
        self.tr1=transLayer(inMap, outMap,dropout)
        
        inMap=outMap
        self.model2=addLayers(inMap,growth_rate,bNeck,layers,dropout)
        inMap+=layers*growth_rate
        
        '''
        For adding more blocks repeat above code i.e
    
        -----------------------------------------------------------------------
        if compression:
            outMap=int(math.floor(inMap*compression))
            
        else:
            outMap=inMap
            
        
        self.tr=transLayer(inMap, outMap,dropout)
        
        
        inMap=outMap
        self.model3=addLayers(inMap,growth_rate,bNeck,layers,dropout)
        inMap+=layers*growth_rate
        
        ------------------------------------------------------------------------
        ''' 
        self.bn=nn.BatchNorm2d(inMap)
        
        self.linear=nn.Linear(inMap,10)
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                size = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / size))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
            elif isinstance(module, nn.Linear):
                module.bias.data.zero_()
        
    def forward(self, x):
        
        x=self.conv1(x)
        
        x=self.model(x)
        x=self.tr(x)
        
        x=self.model1(x)
        x=self.tr1(x)
        
        x=self.model2(x)
        
        x = torch.squeeze(F.avg_pool2d(F.relu(self.bn(x)),8))

        x=self.linear(x)

        x= F.log_softmax(x)

        
        
        return x
        



