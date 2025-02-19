import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt    

class CNNClassification(nn.Module):
    def __init__(self):
        self.epochs=0
        super().__init__()

        self.network = nn.Sequential(
            
            nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16,32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(4,4),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(-1 ,2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
    
    def forward(self, xb):
        return self.network(xb)
        
    @torch.no_grad()
    def inferenzSet(self,dataset):
        self.eval()
        images=[sublist[0] for sublist in dataset]
        images=torch.stack(images)

        labels=[sublist[1] for sublist in dataset]        
        labels=torch.tensor(labels)

        res=self(images)
        _,preds = torch.max(res, dim=1)

        data=list(zip(images,preds.tolist(),labels.tolist()))
        data2_0=[item[0] for item in data if (item[1]!=item[2] and item[2]==0) ]
        data2_1=[item[0] for item in data if (item[1]!=item[2] and item[2]==1) ]
        _,ax = plt.subplots(figsize = (4,int(len(data)/4+0.5)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(data2_0+data2_1,nrow=16).permute(1,2,0))
        
        print("Erg: "+str(torch.sum(preds == labels).item() / len(data)))
        print("ErgC: "+str(torch.sum(preds == labels).item() ))
        print("Erg0: "+str(len(data2_0)))
        print("Erg1: "+str(len(data2_1)))
        plt.show()

    def inferenzImages(self,dataset,start,length=1):
        with torch.no_grad():
            for i in range(start,start+length):
                img,label = dataset[i]
                res=self(img[None,:,:,:])
                _,pred = torch.max(res, dim=1)
                print("Index:"+str(i)+" Predicted class: ",pred[0].item()," Defined class:",label)


    def trainStart(self,epochs, lr, train_loader, opt_func = torch.optim.Adam):
        optimizer = opt_func(self.parameters(),lr)
        self.train()
        for epoch in range(epochs):        
            train_losses = []
            for batch in train_loader:
                loss = self.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                self.eval()
                outputs = [self.validation_step(batch) for batch in train_loader]

                batch_losses, batch_accs = zip(*outputs)
                epoch_loss = torch.stack(batch_losses).mean().item()
                epoch_acc = torch.stack(batch_accs).mean().item()

                print("Epoch "+str(epoch+self.epochs)+", loss: "+str(epoch_loss)+", acc: "+str(epoch_acc))
                if(epoch_acc==1): 
                    break
        self.epochs=self.epochs+epochs
                
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc=torch.tensor(torch.sum(preds == labels).item() / len(preds))                
        return (loss.detach(), acc)

def showImage(dataSet,index):
    img,label = dataSet[index]
    print(f"Label : {dataSet.classes[label]}")
    print(img.shape,label)
    plt.imshow(img.permute(1,2,0))
    plt.show()

def showBatch(dataset,index):
    batchImages=[images for images,_ in dataset]
    images=batchImages[index]
    _,ax = plt.subplots(figsize = (16,12))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
    plt.show()

if __name__ == '__main__':

     # Datensatzpfade (lokaler Speicherort)
    data_dir = "../Bilder/Cashews/Lernen"
    test_data_dir = "../Bilder/Cashews/Test"
       
    trans = [
        #transforms.Resize((40,40)),
        #transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ]

    train_dataset = ImageFolder(data_dir,transform = transforms.Compose(trans))
    test_dataset = ImageFolder(test_data_dir,transforms.Compose(trans))

    print(train_dataset)
    print("Classes in dataset: ",train_dataset.classes)
    print("Classe  in test dataset: ",test_dataset.classes)

    train_dl = DataLoader(train_dataset, batch_size=128, shuffle = True, num_workers = 4, pin_memory = True)

    #showBatch(train_dl,0)
    #showBatch(train_dl,1)
    #showImage(train_dataset,0)

    model = CNNClassification()
    try:
        #model.load_state_dict(torch.load("model.state"))
        model=torch.load("model.state")
    except:
        print("No model found") 

    trainEpochs = 30
    if trainEpochs>0:
        model.trainStart(trainEpochs, 0.001, train_dl)
        #torch.save(model.state_dict(), "model.state")
        torch.save(model, "model.state")

    model.inferenzSet(test_dataset)
    #model.inferenzImages(test_dataset,int(len(test_dataset)/2),2)
    #model.inferenzImages(test_dataset,0,len(test_dataset)-1)