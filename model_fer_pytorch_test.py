# IMPORTANT POUR LE FONCTIONNEMENT DU FER_PYTORCH__TEST.PY 
# !!! NE PAS SUPPRIMEZ !!!
import torch
import torch.nn as nn
import torchvision
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class Model_base(torch.nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)                 
        loss = torch.nn.functional.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)                    
        loss = torch.nn.functional.cross_entropy(out, labels)   
        acc = accuracy(out, labels)         
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()     
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_chnl, out_chnl, padding=1):
    layers = [
        torch.nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=padding, stride=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_chnl),
        torch.nn.MaxPool2d(2),
        torch.nn.Dropout(0.4)]
    return torch.nn.Sequential(*layers)

class Model(Model_base):
   def __init__(self, in_chnls, num_cls):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_chnls, 256, kernel_size=3, padding=1) # 256x48x48
        self.block1 = conv_block(256, 512)           # 512x24x24 
        self.block2 = conv_block(512, 384)                # 384x12x12       
        self.block3 = conv_block(384, 192)       # 192x6x6 
        self.block4 = conv_block(192, 384)       # 384x3x3
    
        self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                        torch.nn.Linear(3456, 256),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(256),
                                        torch.nn.Dropout(0.3),
                                        torch.nn.Linear(256, num_cls))    
   def forward(self, xb):
        out = self.conv1(xb)
        out = self.block1(out)
        out = self.block2(out)       
        out = self.block3(out)
        out = self.block4(out)
        
        return self.classifier(out)



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

index_labelled = np.arange(0, 10, 1)
print(index_labelled)