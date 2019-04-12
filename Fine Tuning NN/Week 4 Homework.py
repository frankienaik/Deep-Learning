
# coding: utf-8

import xml.etree.ElementTree as ET
#from sets import Set
def parsesynsetwords(filen):

  synsetstoclassdescriptions={}
  indicestosynsets={}
  synsetstoindices={}
  ct=-1
  with open(filen) as f:
    for line in f:
      if (len(line)> 5):
        z=line.strip().split()
        descr=''
        for i in range(1,len(z)):
          descr=descr+' '+z[i]
        
        ct+=1
        indicestosynsets[ct]=z[0]
        synsetstoindices[z[0]]=ct
        synsetstoclassdescriptions[z[0]]=descr[1:]
  return indicestosynsets,synsetstoindices,synsetstoclassdescriptions


def test_parsesyn():
  filen = 'synset_words.txt'
  indicestosynsets,synsetstoindices,synsetstoclassdescr=parsesynsetwords(filen)
  clsdict=get_classes()
  
  '''
  for keyval in indicestosynsets.items():
    print(type(keyval[0]),keyval[0],keyval[1])
  '''
  
  
  
  for i in range(1000):
    n1=synsetstoclassdescr[indicestosynsets[i]]
    n2=clsdict[i]
    
    if(n1!=n2):
      print (i)
      print ('n1', n1,'n2:',n2 )

def testparse():
  nm='ILSVRC2012_bbox_val_v3/val/ILSVRC2012_val_00049999.xml'

  tree = ET.parse(nm)
  root = tree.getroot()
  
  '''
  for child in root:
    #print child.tag,'|'
    if child.tag=='object':
      for el in child:
        if el.tag=='name':
          print el.text
  '''        
  for obj in root.findall('object'):
     for name in obj.findall('name'):
       print (name.text)
  #for         
  
def parseclasslabel(nm,synsetstoindices):  
  tree = ET.parse(nm)
  root = tree.getroot()

  lbset=set()
  
  for obj in root.findall('object'):
     for name in obj.findall('name'):
       #print name.text
       ind=synsetstoindices[name.text]
       firstname=name.text
       lbset.add(ind)
       
  if len(lbset)!=1:
    print     ('ERR: len(lbset)!=1',  len(lbset))
    exit()
    
  for s in lbset:
    label=  s
  return label,firstname
  
  
def test_parseclasslabel():
  filen='synset_words.txt'

  
  nm='ILSVRC2012_bbox_val_v3/val/ILSVRC2012_val_00049999.xml'
  
  indicestosynsets,synsetstoindices,synsetstoclassdescr=parsesynsetwords(filen)
  
  label,firstname=parseclasslabel(nm,synsetstoindices)
  
  print(label,firstname,  synsetstoclassdescr[indicestosynsets[label]] )



import getimagenetclassesNewV2 as gic
from skimage import data, io, filters
import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import skimage.color
import cv2
from torch import nn
import torch

from mxnet.gluon.model_zoo import vision

class imagenetDataset(Dataset):
    
    def __init__(self,path_sysnet,path_image,path_labels,transform = None):
        self.path_sysnet = path_sysnet
        self.path_labels = path_labels
        self.path_image = path_image
        self.sysnet_list =[]
        self.image_list=[]
        self.transformations = transform
        self.rgbtransform = transforms.Compose(transforms.Grayscale(3))
#         self.transformations = transforms.Compose([transforms.Resize(224),
#                                                   transforms.CenterCrop(224),
#                                                   transforms.Grayscale(1),
#                                                    transforms.ToTensor()])


    def __len__(self):
        return 2500
    
    def __getitem__(self,index):
        filen=self.path_sysnet + 'synset_words.txt'
        nm= self.path_labels + 'ILSVRC2012_val_'+str(index+1).zfill(8)+'.xml'
        img_name = self.path_image + 'ILSVRC2012_val_'+str(index+1).zfill(8)+'.jpeg' 
        
        indicestosynsets,synsetstoindices,synsetstoclassdescr=gic.parsesynsetwords(filen)
        image = Image.open(img_name)
        image = image.convert('RGB')
#         image = image.convert('L')
#         image = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_BGR2GRAY)
#         print(image.size)
#         print(image.ndim)
#         Image.convert('RGB',image)
#         print(Image.getchannel('R'))
#         print(image.size)
#         if(image.shape[0] == 1):
#             image = self.rgbtransform(image)
        

#         print(image.shape)

#             img = image.convert('LA')
#             image = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_BGR2GRAY)
#             skimage.color.rgb2gray(image)
#             image = rgb2gray(image)

        
#         image = self.transformations(img)
        
#         print(img_name)
#         if (image.shape[0]!=1):
#             print(image.shape)
        label,firstname=gic.parseclasslabel(nm,synsetstoindices)
#         self.sysnet_list.append((label,firstname,  synsetstoclassdescr[indicestosynsets[label]] ))
#         self.image_list.append(image)
        
        labels = (label,firstname,  synsetstoclassdescr[indicestosynsets[label]])
        
        if self.transformations:
            image = self.transformations(image)
        
#         print(labels.shape)
#         result = {'image':image,'labels':(label,firstname,  synsetstoclassdescr[indicestosynsets[label]])}
        
#         return result
#         print('next')
#         print(image)
#         if (torch.all(torch.eq(image[0], image[1]))):
#             print(img_name)
        return image,labels

'''   
    def getitem(self):
            
        filen=self.path_sysnet + 'synset_words.txt'
        for num in range(1,2501):
            nm= self.path_labels + 'ILSVRC2012_val_'+str(num).zfill(8) +'.xml'
            img_name = self.path_image + 'ILSVRC2012_val_'+str(num).zfill(8)+'.jpeg' 
            
            indicestosynsets,synsetstoindices,synsetstoclassdescr=gic.parsesynsetwords(filen)
            image = io.imread(img_name)
            
            label,firstname=gic.parseclasslabel(nm,synsetstoindices)
            self.sysnet_list.append((label,firstname,  synsetstoclassdescr[indicestosynsets[label]] ))
            self.image_list.append(image)
            
            return self.sysnet_list,self.image_list
'''


# ## Problem 1
# 



no_classes = 1000
def test(model, device, test_loader):
    model.train(mode=False)
    model.eval()
    confusion_matrix=torch.zeros(no_classes,no_classes)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i ,(data, target) in enumerate(test_loader):
            data = data.to(device)#, target.to(device)
            labels = target[0].to(device)
            
            if (len(data.shape) == 5): #if it is fivecrop
                bs, ncrops, c, h, w = data.size()
                data = data.view(-1,c,h,w)
                output = model(data)
                output = output.view(bs,ncrops,-1).mean(1)
#             labels = torch.unsqueeze(labels,1).shape
            else:
                output = model(data)
    
            
#             print(output)
#             print(data.shape)
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
#             print(pred.shape)
#             print(torch.unsqueeze(labels,1).shape)
            
            for t,p in zip(labels.view(-1),pred.view(-1)):
                confusion_matrix[t.long(),p.long()] +=1
            
            correct += pred.eq(labels.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return confusion_matrix, accuracy


class Squeezenet_330(nn.Module):
    def __init__(self, num_classes=no_classes):
        super(Squeezenet_330,self).__init__()
        self.num_classes = num_classes
        self.avgPool = nn.AdaptiveAvgPool2d((224,224))
        self.trained_model = models.squeezenet1_1(pretrained = True)
    def forward(self,x):
        x = self.avgPool(x)
        x = self.trained_model.features(x)
        x = self.trained_model.classifier(x)
        return x.view(x.size(0), self.num_classes)
    
class Inceptionnet_330(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=no_classes):
        super(Inceptionnet_330, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d((224,224))
        self.trained_model = models.inception_v3(pretrained=True)
    def forward(self, x):
        x = self.avgPool(x)
        x = self.trained_model.forward(x)
        return x
    




import torchvision.models as models


def main():
    sysnet_path = ''
    image_path = 'imagenet2500/imagespart/'
    label_path = 'ILSVRC2012_bbox_val_v3/val/'

    device = torch.device("cpu")
#     resnet18 = vision.resnet18_v1(pretrained=True)
#     squeezenet = vision.squeezenet1_0(pretrained = True)
    inception = models.inception_v3(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)
    
    
    print("Problem 1:")
    #Problem 1:
    image_datasetQ1_norm = imagenetDataset(sysnet_path,image_path,label_path,transforms.Compose([transforms.Resize(224),
                                                      transforms.CenterCrop(224),
    #                                                         transforms.Grayscale(1),
                                                        transforms.ToTensor(),   
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                   )
    image_datasetQ1 = imagenetDataset(sysnet_path,image_path,label_path,transforms.Compose([transforms.Resize(224),
                                                      transforms.CenterCrop(224),
    #                                                         transforms.Grayscale(3),
                                                            transforms.ToTensor()]  )
                                     )

#     print(image_datasetQ1.__getitem__(0))
#     print(image_datasetQ1_norm.__getitem__(0))
    
    image_dataset_loaderQ1_norm = torch.utils.data.DataLoader(dataset=image_datasetQ1_norm,
                                                        batch_size=10,
                                                        shuffle=False)

    image_dataset_loaderQ1 = torch.utils.data.DataLoader(dataset=image_datasetQ1,
                                                        batch_size=10,
                                                        shuffle=False)
    
    #not normalized
    _,accuracyQ1 = test(squeezenet,device,image_dataset_loaderQ1)
    #normalized
    
    print("Accuracy for unnormalized data: ", accuracyQ1)
    
    _,accuracyQ1_norm = test(squeezenet,device,image_dataset_loaderQ1_norm)
#     for data , target in image_dataset_loaderQ1:
#         print(data)
#         break
#     for data , target in image_dataset_loaderQ1_norm:
#         print(data)
#         break
    print("Accuracy for normalized data: ", accuracyQ1_norm)
    
    print("-" * 50)
    print("Problem 2")
    # problem 2:
    image_datasetQ2 = imagenetDataset(
        sysnet_path,image_path,label_path,transforms.Compose(
            [transforms.Resize(280),
            transforms.FiveCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(norm) for norm in norms]))])

#             (lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(), 
#                                                             transforms.Normalize(
#                                                                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                                                             )])(crop) for crop in crops]))]  
        )
    
    image_dataset_loaderQ2 = torch.utils.data.DataLoader(dataset=image_datasetQ2,
                                                        batch_size=10,
                                                        shuffle=False)    

    _,accuracyQ2 = test(squeezenet,device,image_dataset_loaderQ2)
                        
    print("Accuracy for FiveCrop data: ", accuracyQ2)
    
    print("-" * 50)
    print("Problem 3:")
    # problem 3:
    image_datasetQ3 = imagenetDataset(
        sysnet_path,image_path,label_path,transforms.Compose(
            [transforms.Resize(330),
            transforms.FiveCrop(330),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(norm) for norm in norms]))])

#             (lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(), 
#                                                             transforms.Normalize(
#                                                                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                                                             )])(crop) for crop in crops]))]  
        )
    
    image_dataset_loaderQ3 = torch.utils.data.DataLoader(dataset=image_datasetQ3,
                                                        batch_size=10,
                                                        shuffle=False)    
    

    _,accuracyQ3 = test(Squeezenet_330(),device,image_dataset_loaderQ3)
    
    print("Accuracy for FiveCrop data (squeezenet): ", accuracyQ3)

    _,accuracyQ3_2 = test(Inceptionnet_330(),device,image_dataset_loaderQ3)
    
    print("Accuracy for FiveCrop data (inception): ", accuracyQ3_2)
#     print("Accuracy for FiveCrop data: ", accuracyQ2)
if __name__ == '__main__':
    main()

