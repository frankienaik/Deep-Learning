#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 02:43:22 2019

@author: albertsuryanto
"""

import pandas as pd
import os
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import time
from statistics import mean
import matplotlib.pyplot as plt


class PascalVOC:
    """
    Handle Pascal VOC dataset
    """
    def __init__(self, root_dir):
        """
        Summary: 
            Init the class with root dir
        Args:
            root_dir (string): path to your voc dataset
        """
        self.root_dir = root_dir
        self.img_dir =  os.path.join(root_dir, 'JPEGImages/')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')


    def list_image_sets(self):
        """
        Summary: 
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    def _imgs_from_category(self, cat_name, dataset):
        """
        Summary: 
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(self.set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', cat_name])
        return df

    
    def _imgs_df(self, dataset):
        """
        Summary: 
        Args:
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(self.set_dir, dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename'])
        return df
    
    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Summary: 
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "trainval", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name, dataset)
        df = df[df['true'] == 1]
        return df['filename'].values


        
class VOCDataset(Dataset):
    '''trainpath, testpath, valpath, valtestpath, label_path,'''
    def __init__(self, path, image_label_df, transform = None):
        self.image_label_df = image_label_df
        self.transformations = transform   
        self.path = path
        
    def __len__(self):
        return(int(self.image_label_df.shape[0]))
        #return 10
        
    def __getitem__(self,index):  
        imagename, label = self.image_label_df.loc[index].filename, self.image_label_df.loc[index].one_hot_encoding

        imagename = imagename + '.jpg'
        image = Image.open(self.path + 'JPEGImages/' + imagename)

        if self.transformations:
            imagetrans = self.transformations(image)

        return imagetrans, torch.Tensor(label),imagename




def obtainLoader(path, file_list, batch_size, transform):
    imagelabel = VOCDataset(path, file_list, transform)  
    imagelabel_loader = torch.utils.data.DataLoader(dataset=imagelabel,
                                                    batch_size=batch_size, shuffle=True)

    return imagelabel_loader




def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train(mode = True).to(device)
    start_time = time.time()
    train_loss = 0
    
    n_batches = len(train_loader)
    print_every = n_batches // log_interval
    
    for batch_idx, (data, target, imagename) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        if (len(data.shape) == 5): #if it is fivecrop or tencrop
                bs, ncrops, c, h, w = data.size()
                data = data.view(-1,c,h,w)
                output = model(data)
                output = output.view(bs,ncrops,-1).mean(1)
        
        else:
                output = model(data)

        loss = F.binary_cross_entropy(F.sigmoid(output),target)
        train_loss += loss.item() # sum up batch loss
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % (print_every + 1) == 0:
            print('Train Epoch: {}, {:d}% \tLoss: {:.6f} \t took: {:.2f}s'.format(
                epoch,
                int(100 * (batch_idx + 1) / n_batches),
                train_loss / n_batches,
                time.time() - start_time))
            
    print("Total Loss: {:.6f}".format(train_loss / n_batches))
    return train_loss






def test(model, device, test_loader):         
    model.train(mode=False).to(device)
    model.eval().to(device)
    start_time = time.time()
    test_loss = 0
    dict_output = {}
    
    with torch.no_grad():
        for i ,(data, target, imagename) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            
            if (len(data.shape) == 5): #if it is fivecrop or tencrop
                bs, ncrops, c, h, w = data.size()
                data = data.view(-1,c,h,w)
                output = model(data)
                output = output.view(bs,ncrops,-1).mean(1)

            else:
                output = model(data)
            
            sigmoid_output = F.sigmoid(output)
            test_loss += F.binary_cross_entropy(sigmoid_output, target).item() 
            
            # making output to a one hot vectore:
            for i in range(len(imagename)):
                dict_output[imagename[i]] = {}
                dict_output[imagename[i]]["sigmoid prediction"] = sigmoid_output[i]
                dict_output[imagename[i]]["true value"] = target[i]
        
    test_loss /= len(test_loader.dataset)

    print("Total time: {:.2f}s".format(time.time() - start_time))
    print("Total loss: {:.6f}s".format(test_loss))
    
    return test_loss, dict_output



def creating_dataframe_dataset(path, pv, cat_list, phase):
    df = pv._imgs_df(phase)
    
    for category in cat_list:          
        category_df = pv._imgs_from_category(category, phase)
        df = pd.merge(df, category_df, left_on='filename', right_on='filename', how = "inner")
    
    df = df.replace(-1,0)
    df['one_hot_encoding'] = df[df.columns[1:]].values.tolist()

    return df




def train_and_validate_model(path, training_df, validation_df, cat_list, transform):
    
    if transform[1] == 1: #fivecrop_transformation:
        name = 'model_5crops.pt'
    else:
        name = 'model_rotation.pt'
        
    least_loss = 10000 #best loss
    train_loss_list =[]
    validation_loss_list = []

    epochs = 20
    batch_size = 16
    log_interval= 10

    lr = 0.01
    momentum = 0.9

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = models.resnet18(pretrained=True).to(device)

    lt = 8
    cntr = 0

    for child in model.children(): 
        cntr += 1
        if cntr < lt: 
            for param in child.parameters():
                param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 20)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    imagelabel_train_loader = obtainLoader(path, training_df, batch_size, transform[0])
    imagelabel_val_loader = obtainLoader(path, validation_df, batch_size, transform[0])

    for epoch in range(1, epochs + 1):
        print("Training", epoch)
        train_loss = train(log_interval, model, device, imagelabel_train_loader, optimizer, epoch)
        train_loss_list.append(train_loss)

        print("\nValidation", epoch)
        validation_loss, _ = test(model, device, imagelabel_val_loader)                      
        validation_loss_list.append(validation_loss)

        if validation_loss < least_loss:
            least_loss = validation_loss

            print("\nBest loss: {:.5f}".format(least_loss))
            print('Save Model')
            if epoch == epochs:
                save_model_name = path + name
            else:
                save_model_name = path + name + str(epoch)
            torch.save(model.state_dict(), save_model_name)

    print('-'*50)    
    return train_loss_list, validation_loss_list




def loss_graph(loss_list, phase):
    plt.title(phase + " loss over epoch")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(0, len(loss_list)), loss_list)
    plt.show()  



def prediction_output_with_best_model(image_path, pretrained_model_name_path, test_df, transform):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = models.resnet18(pretrained=False).to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 20)
    
    model.load_state_dict(torch.load(pretrained_model_name_path, map_location=device))


    imagelabel_test_loader = obtainLoader(image_path, test_df, 1, transform)

    _, dict_output = test(model, device, imagelabel_test_loader)                   
    return dict_output




def dataframe_output(dict_output, cat_list):
    for filename, pred_and_true in dict_output.items():
        for label_col, cat_sigmoid in enumerate(pred_and_true['sigmoid prediction']):
            category = cat_list[label_col]

            dict_output[filename]['pred sigmoid ' + category] = cat_sigmoid.cpu().numpy()

        for label_col, true_one_hot_encoding in enumerate(pred_and_true['true value']):
            category = cat_list[label_col]

            dict_output[filename]['true one hot ' + category] = true_one_hot_encoding.cpu().numpy()

    df = pd.DataFrame.from_dict({(i): dict_output[i] for i in dict_output.keys()}, orient='index')
    
    return df
    
    
def max_threshold_each_class(df_output, cat_list):
    max_threshold_each_class_dict = {}

    for label_col in range(len(cat_list)):
        category = cat_list[label_col]

        max_threshold_each_class_dict[category] = df_output['pred sigmoid ' + category].max()

    return max_threshold_each_class_dict
       

def average_precision(df_output, cat_list, max_threshold_across_all_classes):
    average_precisions = []

    for i in np.arange(0, max_threshold_across_all_classes, max_threshold_across_all_classes/20.0): 
        AP = 0

        for label_col in range(len(cat_list)):
            category = cat_list[label_col]

            df_output['pred one hot {}'.format(category)] = np.where(df_output['pred sigmoid ' + category] > i, 1, 0)
            df_output['TP {}'.format(category)] = np.where((df_output['pred one hot {}'.format(category)] == 1) & (df_output['true one hot {}'.format(category)] == 1), 1, 0) 

            sub_df = df_output[['pred sigmoid ' + category, "TP " + category]]
            sub_df = sub_df.sort_values(by=['pred sigmoid ' + category], ascending=False)
            sub_df["Rank"] = range(1, len(df_output)+1)
            sub_df["All_Positives"] = df_output['true one hot ' + category].sum()
            sub_df["Precision"] = sub_df["TP " + category].cumsum() / sub_df["Rank"]
            sub_df["Recall"] = sub_df["TP " + category].cumsum() / sub_df["All_Positives"]
#            sub_df["Recall"] = sub_df["TP " + category].cumsum()


            recall_arr = sub_df["Recall"].values
            precision_arr = sub_df["Precision"].values

            full_dict = dict.fromkeys(set(recall_arr))
            recall_precision_pairing = dict.fromkeys(set(recall_arr))
            standard_recalls = np.arange(0, 1.1, 1/10.0)
            standard_recall_precision_pairing = dict.fromkeys(standard_recalls)

            for key in full_dict.keys():
                full_dict[key] = []

            for j in range(len(recall_arr)):
                full_dict[recall_arr[j]].append(precision_arr[j])
                recall_precision_pairing[recall_arr[j]] = max(full_dict.get(recall_arr[j]))

            for standard_recall in standard_recall_precision_pairing.keys():
                candidates = []
                for recall in recall_precision_pairing.keys():
                    if recall >= standard_recall:
                        candidates.append(recall_precision_pairing.get(recall))
                if len(candidates) == 0:
                    candidates.append(0.0)
                standard_recall_precision_pairing[standard_recall] = max(candidates)

            AP += sum(standard_recall_precision_pairing.values()) / 11

        average_precisions.append(AP/20)

    return average_precisions


def AP_graph(average_precision_list, max_threshold_across_all_classes):
    plt.title("Average Precision Across Recall")
    plt.xlabel('Recall')
    plt.ylabel('Average Precision Score')
    plt.plot(np.arange(0, max_threshold_across_all_classes, max_threshold_across_all_classes/20.0), average_precision_list)
    plt.show()  
   

def tail_accuracy_each_class(df_output, max_threshold_dict, max_threshold_each_class_dict):
    class_dict_threshold = {}

    for category, max_threshold in max_threshold_each_class_dict.items():
        class_dict_threshold[category] = {}

        for i in np.arange(0, max_threshold, max_threshold/20.0): 
            df_output['pred one hot {}'.format(category)] = np.where(df_output['pred sigmoid ' + category] > i, 1, 0)

            sub_df = df_output[['pred sigmoid ' + category, "pred one hot " + category, 'true one hot ' + category]]
            sub_df = sub_df.sort_values(by=['pred sigmoid ' + category], ascending=False).head(50)

            tp = len(sub_df.loc[(sub_df['pred one hot ' + category] == 1) & (sub_df['true one hot ' + category] == 1)])
            fp = len(sub_df.loc[(sub_df['pred one hot ' + category] == 1) & (sub_df['true one hot ' + category] == 0)])

            if ((tp + fp) == 0):
                precision = 0
            else:
                precision = tp / (tp + fp)

            class_dict_threshold[category][i] = precision

        mean_average_precision_per_class = mean(class_dict_threshold[category].values())
        print('Average Precision for {}: {}'.format(category, mean_average_precision_per_class))
                
    return class_dict_threshold


def tail_accuracy_graph(tail_accuracy_dict, cat_list):
    for label_col in range(len(cat_list)):
        category = cat_list[label_col] 
        lists = sorted(tail_accuracy_dict[category].items())

        x, y = zip(*lists)

        plt.xlabel('Threshold')
        plt.ylabel('Tail Accuracy Score')

        plt.plot(x, y)

        plt.title("Prediction for class {}".format(category))
        plt.show()


def main():
    #path = '/Users/albertsuryanto/SUTD/Term 7/Deep Learning/Project/pascalvoc/VOCdevkit/VOC2012/'
    path = ''
    pv = PascalVOC(path)
    cat_list = pv.list_image_sets()
    
    training_df = creating_dataframe_dataset(path, pv, cat_list, 'train')
    validation_df = creating_dataframe_dataset(path, pv, cat_list, 'val')

    ###### FiveCrop
    fivecrop_transformation = [transforms.Compose([
        transforms.Resize(280),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(norm) for norm in norms]))
    ]),1]
    
    ###### Rotation and CenterCrop
    rotation_transformation = [transforms.Compose([
        transforms.RandomRotation((-15,15)),
        transforms.Resize(280),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),2]
    
    ##### training for fivecrop
    train_loss_list, val_loss_list = train_and_validate_model(path, training_df, validation_df, cat_list, fivecrop_transformation)
    loss_graph(train_loss_list, 'train')     
    
    ##### training for rotation
    train_loss_list, val_loss_list = train_and_validate_model(path, training_df, validation_df, cat_list, rotation_transformation)
    loss_graph(train_loss_list, 'train') 
    
    # FiveCrop model
    print('-'*50)
    print("FiveCrop Model")
    #pretrained_model_name_path = '/Users/albertsuryanto/SUTD/Term 7/Deep Learning/Project/pascalvoc/VOCdevkit/VOC2012/trained_fivecrop_model.pt'
    pretrained_model_name_path = path + 'model_5crops.pt'
    
    dict_output = prediction_output_with_best_model(path, pretrained_model_name_path, validation_df, fivecrop_transformation[0])
    df_output = dataframe_output(dict_output, cat_list)

    max_threshold_dict = max_threshold_each_class(df_output, cat_list)
    max_threshold_across_all_classes = max(max_threshold_dict.values())

    average_precision_list = average_precision(df_output, cat_list, max_threshold_across_all_classes)
    AP_graph(average_precision_list, max_threshold_across_all_classes)

    tail_accuracy_dict = tail_accuracy_each_class(df_output, max_threshold_dict, max_threshold_dict)
    tail_accuracy_graph(tail_accuracy_dict, cat_list)


    # Rotation model
    print('-'*50)
    print("Rotation Model")

    #pretrained_model_name_path = '/Users/albertsuryanto/SUTD/Term 7/Deep Learning/Project/pascalvoc/VOCdevkit/VOC2012/trained_rotation_model.pt'
    pretrained_model_name_path = path + 'model_rotation.pt'
    dict_output = prediction_output_with_best_model(path, pretrained_model_name_path, validation_df, rotation_transformation[0])
    df_output = dataframe_output(dict_output, cat_list)

    max_threshold_dict = max_threshold_each_class(df_output, cat_list)
    max_threshold_across_all_classes = max(max_threshold_dict.values())

    average_precision_list = average_precision(df_output, cat_list, max_threshold_across_all_classes)
    AP_graph(average_precision_list, max_threshold_across_all_classes)

    tail_accuracy_dict = tail_accuracy_each_class(df_output, max_threshold_dict, max_threshold_dict)
    tail_accuracy_graph(tail_accuracy_dict, cat_list)


if __name__ == '__main__':
    main()
    
