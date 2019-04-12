# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:16:38 2019

@author: Frank
"""

import operator 
import math
import json
import tkinter as tk                # python 3
from tkinter import font  as tkfont # python 3
from tkinter import filedialog
#import Tkinter as tk     # python 2
#import tkFont as tkfont  # python 2

import pandas as pd
import os
#from bs4 import BeautifulSoup
#from more_itertools import unique_everseen
import numpy as np
#import matplotlib.pyplot as plt
#import skimage
#from skimage import io
from skimage import data, io, filters
# import cv2
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageTk
import skimage.color
#import cv2
from torch import nn
import torch

import torch.optim as optim
import torch.nn.functional as F

import torchvision.models as models

'''
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
'''

device = torch.device("cpu")


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
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        self.cache_dir = os.path.join(root_dir, 'csvs')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

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
            names=['filename', 'true'])
        return df

    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Summary: 
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name, dataset)
        df = df[df['true'] == 1]
        return df['filename'].values
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

global currentpage
global pagecount
currentpage = 1
pagecount = 5 #read a val.shape[0]/20 
    
    
    
    
    


class SampleApp(tk.Tk):
    
    
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
        
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        #self.filepath= tk.StringVar()
        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        

        self.frames = {}
        for F in (StartPage, PageSubmit, PageSeeVal):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")
        
        
        self.show_frame("StartPage")
        

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


    

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Welcome to our simple GUI for the SMALL Project (:", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(self, text="Submit Image",
                            command=lambda: controller.show_frame("PageSubmit"))
        button2 = tk.Button(self, text="View Precomputed Scores",
                            command=lambda: controller.show_frame("PageSeeVal"))
        button1.pack()
        button2.pack()



class PageSubmit(tk.Frame):
            
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Choose a model", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        our_models = [
            ("FiveCrop",1),
            ("Rotation",2)
        ]
        model_number = tk.IntVar()
        model_number.set(1)  # initializing the choice, i.e. Python
        for model,val in (our_models):
            tk.Radiobutton(self, 
                          text=model,
                          padx = 20, 
                          variable=model_number, 
                          value=val).pack(anchor=tk.W)
            
        scrollbar = tk.Scrollbar(self)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(self, yscrollcommand=scrollbar.set)
        '''
        for i in range(1000):
            listbox.insert(tk.END, str(i))

        '''
        panel = tk.Label(self)
        '''
        imagecanvas = tk.Canvas(self, 
           width=200, 
           height=200)
        
        imagecanvas.pack()
        
        '''

        
        button1 = tk.Button(self, text="Choose Image",
                           command=lambda: self.chooseImageAndRunModel(model_number,panel,listbox))#lambda: controller.show_frame("StartPage"))
        

        
        
        button2 = tk.Button(self, text="Back",
                           command=lambda: controller.show_frame("StartPage"))
        button1.pack(anchor=tk.W)
        button2.pack(anchor=tk.W)
        listbox.pack(anchor=tk.W, fill=tk.BOTH,expand = "yes")
        
        scrollbar.config(command=listbox.yview)
        
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        
        

        
    def chooseImageAndRunModel(self,model_number,panel,listbox):
        #file_path = tk.StringVar()
        print(model_number.get())
        file_path = tk.filedialog.askopenfilename()
        img = Image.open(file_path)
        image = ImageTk.PhotoImage(img)
        
        #imagecanvas.configure.create_image(0,0, anchor=tk.NW,image=img)
        #self.
        panel.configure(image = image)
        panel.photo = image
        
        #### THIS IS THE PREDICTION
        
        
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        
        model3 = models.resnet18(pretrained=False).to(device)
        num_ftrs = model3.fc.in_features
        model3.fc = nn.Linear(num_ftrs, 20)
        
        if model_number.get() == 1: # five crop
            transform = transforms.Compose(
                        [transforms.Resize(280),
                        transforms.FiveCrop(224),
                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                        transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(norm) for norm in norms]))
                        ])
            model_path = "model_5crops.pt"
            model3.load_state_dict(torch.load(model_path, map_location=device))
        else: #rotation
            transform = transforms.Compose(
                    [transforms.RandomRotation((-15,15)),
                    transforms.Resize(280),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            #         ,transforms.ToPILImage()
                    ])
                   
            ### CHANGE THIS ###
            model_path = "model_rotation.pt"
            model3.load_state_dict(torch.load(model_path, map_location=device))
        
        #path = '/content/drive/My Drive/Colab Notebooks/JPEGImages/'
        
        #imagename = '2007_000027.jpg'
        #image = Image.open(path + imagename)
        

        
        imagetrans = transform(img)
        
        model3.eval().to(device)
        data = imagetrans.to(device)
        
        if (len(data.shape) == 4): #if it is fivecrop or tencrop
            print(data.shape)
            ncrops, c, h, w = data.size()
            data = data.view(-1,c,h,w)
            output = model3(data)
            print('preoutput',output.shape)
            output = output.view(ncrops,-1).mean(0)
            print('output',output.shape)
        else:
            data = data.view(1,data.shape[0],data.shape[1],data.shape[2])
            output = model3(data)
            output = output.view(-1)
            print('output',output.shape)
        
        
        sigmoid_output = F.sigmoid(output)
        pv=PascalVOC('')
#
        image_set = pv.list_image_sets()
        listbox.delete(0,tk.END)
        outputstr = ''
        sorted_list = []
        for i in range(len(image_set)):
            #print(image_set[i],sigmoid_output[i].item())
            sorted_list.append((image_set[i], sigmoid_output[i].item()))
            #print(type(sigmoid_output[i].item()))
            #outputstr = str(image_set[i]) + ': ' + str(sigmoid_output[i].item()) + '\n'
            #listbox.insert(tk.END,outputstr) 
        #sorted_list = sorted_list.sort(key=operator.itemgetter(1))
        #print(sorted_list)
        sorted_list.sort(key=lambda x: x[1],reverse=True)
        #print(sorted_list)
        for category, prediction in sorted_list:
            outputstr = str(category) + ': ' + str(prediction) + '\n'
            listbox.insert(tk.END,outputstr) 
            
            
class PageSeeVal(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Pick a Model and a Class", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        pv=PascalVOC('')
#
        image_set = pv.list_image_sets()
        
        class_name = tk.StringVar()
        class_name.set(image_set[0])
        

        img_number = tk.IntVar()
        our_models = [
            ("FiveCrop",1),
            ("Rotation",2)
        ]
        model_number = tk.IntVar()
        model_number.set(1)  # initializing the choice, i.e. Python
        for model,val in (our_models):
            tk.Radiobutton(self, 
                          text=model,
                          padx = 20, 
                          variable=model_number, 
                          value=val,command = self.resetCurrentPage2).pack(anchor=tk.W)
        
        '''
        self.button1 = tk.Button(self, text="Show First 20 Results",
                           command=lambda: self.showFirst20Results(class_name,img_number,panel_list))
        self.button1.pack(anchor=tk.W)   
                

        self.button3 = tk.Button(self, text="Next 20",state=tk.DISABLED,
                           command=lambda: self.showNext20(class_name,img_number,panel_list))
        self.button3.pack(anchor=tk.W)   
                
        self.button4 = tk.Button(self, text="Previous 20",state=tk.DISABLED,
                           command=lambda: self.showPrev20(class_name,img_number,panel_list))
        self.button4.pack(anchor=tk.W) 
        '''
        '''
        scrollbar = tk.Scrollbar(self)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(self, yscrollcommand=scrollbar.set)
        
        for i in range(1000):
            listbox.insert(tk.END, str(i))
        
        listbox.pack(anchor=tk.W, fill=tk.BOTH,expand = "yes")
        
        scrollbar.config(command=listbox.yview)
        '''
        
        
        '''
        vscrollbar = tk.Scrollbar(self)

        c= tk.Canvas(self,background = "#D2D2D2",yscrollcommand=vscrollbar.set,scrollregion=(0, 0, 10000, 10000))
        
        vscrollbar.config(command=c.yview)
        vscrollbar.pack(side=tk.RIGHT, fill=tk.Y) 
        
        f=tk.Frame(c) #Create the frame which will hold the widgets
        
        
        
        
        c.pack(anchor=tk.W, fill="both",expand=True)
        
        #Updated the window creation
        c.create_window(0,0,window=f, anchor='nw')
        '''
        
        frame = tk.Frame(self, bd=10, relief=tk.SUNKEN)

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        # xscrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        # xscrollbar.grid(row=1, column=0, sticky=tk.E+tk.W)
        
        self.yscrollbar = tk.Scrollbar(frame)
        self.yscrollbar.grid(row=0, column=1, sticky=tk.N+tk.S)
        
        self.canvas = tk.Canvas(frame, bd=200,
                        #xscrollcommand=xscrollbar.set,
                        yscrollcommand=self.yscrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        
        
        f=tk.Frame(self.canvas)
        
        self.canvas.create_window(0,0,window=f, anchor='nw')
        #Added more content here to activate the scroll
        panel_list = []
        for i in range(40):
           
            panel_list.append(tk.Label(f,wraplength=350))
            
        #print(panel_list)
        
        w = tk.OptionMenu(self, class_name, *image_set,command=self.resetCurrentPage)
        w.pack()
        
            
        self.button1 = tk.Button(self, text="Show First 20 Results",
                           command=lambda: self.showFirst20Results(class_name,img_number,panel_list,model_number))
        self.button1.pack(anchor=tk.W)
                

        self.button3 = tk.Button(self, text="Next 20",state=tk.DISABLED,
                           command=lambda: self.showNext20(class_name,img_number,panel_list,model_number))
        self.button3.pack(anchor=tk.W)   
                
        self.button4 = tk.Button(self, text="Previous 20",state=tk.DISABLED,
                           command=lambda: self.showPrev20(class_name,img_number,panel_list,model_number))
        self.button4.pack(anchor=tk.W)   
         
        for panel in panel_list:
            panel.pack(expand=True)
            
        
        self.button2 = tk.Button(self, text="Back",
                           command=lambda: controller.show_frame("StartPage"))
        self.button2.pack(anchor=tk.W)

        self.update()
        #xscrollbar.config(command=canvas.xview)
        #self.yscrollbar.config(command=self.canvas.yview)
        
        frame.pack()

        
    def resetCurrentPage(self,event):
        print('reset')
        global currentpage, pagecount
        currentpage = 1
        if self.button3["state"] != 'disabled':
            self.button3.config(state=tk.DISABLED)
        if self.button4["state"] != 'disabled':
            self.button4.config(state=tk.DISABLED)
#        button2 = tk.Button(self, text="Submit",
#                           command=lambda: self.)
#        button.pack()
        
    def showFirst20Results(self,class_name,img_number,panel_list,model_number):
        global currentpage, pagecount
        currentpage = 1
        
        if model_number.get() == 1:
            path = 'csv_and_dict_fiveCrop/dictionary/transform1_'
        else:
            path = 'csv_and_dict_rotation/dictionary/'
        
        
        img_number.set(currentpage)
        if self.button3["state"] != 'normal':
            self.button3.config(state=tk.NORMAL)
        if self.button4["state"] != "disabled":
            self.button4.config(state=tk.DISABLED)
        
        total_height = 0
        
        #################
        class_dictionary_path = path + class_name.get() +'_dictionary.json'
        with open(class_dictionary_path) as f:
            data = json.load(f)
        class_name_pred_sigmoid = 'pred sigmoid ' + class_name.get()
        image_sorted_per_class = []
        for key, value in data.items():
            image_sorted_per_class.append((value['Unnamed: 0'],value[class_name_pred_sigmoid]))
        image_path = 'JPEGImages/'
        pagecount = math.ceil(len(image_sorted_per_class)/20)
        
        #image_dictionary =
        #image_list = list(image_dictionary)
        #################
        '''
        file_path = tk.filedialog.askopenfilename()
        img = Image.open(file_path)
        image = ImageTk.PhotoImage(img)
        
        width,height_img = img.size
        
        print(width,height_img)
        '''
        
        for i in range(20):
            ###########
            try:
                img = Image.open(image_path + str(image_sorted_per_class[i][0]))
                image = ImageTk.PhotoImage(img)
                width,height_img = img.size
                total_height += height_img
                ############
                
                a = i*2
                panel_list[a].configure(image = image,height = height_img)
                panel_list[a].photo = image
                panel_list[a+1].configure(text = str(i+1) + ') '+class_name.get()+': ' +str(image_sorted_per_class[(i)][1]),height=2)
            
            except:
                print("reached the end")
                break
            
        y_height = total_height + height_img*3 + 2*20 #height_image is for buffer
        #print(y_height)
        self.canvas.config(scrollregion=(0,0,0,y_height))
        self.update()
        self.yscrollbar.config(command=self.canvas.yview)
        '''
        for panel in panel_list:
            panel.configure(image = image)
            panel.photo = image
        '''
        #imagecanvas.configure.create_image(0,0, anchor=tk.NW,image=img)
        #self.
        '''
        panel.configure(image = image)
        panel.photo = image
        '''
        pass
    
    def showNext20(self,class_name,img_number,panel_list,model_number):
        global currentpage, pagecount

        currentpage += 1
 
        if currentpage >= pagecount:
            self.button3.config(state=tk.DISABLED)
            
        if self.button4["state"] != "normal":
            self.button4.config(state=tk.NORMAL)   # button to page up is enabled
            
        if model_number.get() == 1:
            path = 'csv_and_dict_fiveCrop/dictionary/transform1_'
        else:
            path = 'csv_and_dict_rotation/dictionary/'
        
            
        #file_path = tk.filedialog.askopenfilename()
        #img = Image.open(file_path)
        #image = ImageTk.PhotoImage(img)
        #####
        #path = 'csv_and_dict/dictionary/'
        class_dictionary_path = path + class_name.get() +'_dictionary.json'
        with open(class_dictionary_path) as f:
            data = json.load(f)
        class_name_pred_sigmoid = 'pred sigmoid ' + class_name.get()
        image_sorted_per_class = []
        for key, value in data.items():
            image_sorted_per_class.append((value['Unnamed: 0'],value[class_name_pred_sigmoid]))
        image_path = 'JPEGImages/'
        pagecount = math.ceil(len(image_sorted_per_class)/20)
        #image_dictionary =
        #image_list = list(image_dictionary)

        total_height = 0
        
        for i in range(20):
            try:
                print(str(((currentpage-1)*20)+i+1))
                        
                img = Image.open(image_path + str(image_sorted_per_class[(currentpage-1)*20 + i][0]))
                image = ImageTk.PhotoImage(img)
                width,height_img = img.size
                total_height += height_img
                
                a = i*2
                panel_list[a].configure(image = image,height = height_img)
                panel_list[a].photo = image
                panel_list[a+1].configure(text = str((currentpage-1)*20 + i + 1) +') ' + class_name.get() +': ' + str(image_sorted_per_class[(currentpage-1)*20 + i][1]),height=2)        

            except:
                print("reached the end")
                break
                
        y_height = total_height + height_img*3 + 2*20 #height_image is for buffer
        print(y_height)
        self.canvas.config(scrollregion=(0,0,0,y_height))
        self.update()
        self.yscrollbar.config(command=self.canvas.yview)
        
        img_number.set(currentpage)
        pass
    
    def showPrev20(self,class_name,img_number,panel_list,model_number):
        
        global currentpage, pagecount

        currentpage -= 1 
        
        if currentpage == 1:
            self.button4.config(state=tk.DISABLED)
            
        if self.button3["state"] != "normal":
            self.button3.config(state=tk.NORMAL)   # button to page down is enabled


        if model_number.get() == 1:
            path = 'csv_and_dict_fiveCrop/dictionary/transform1_'
        else:
            path = 'csv_and_dict_rotation/dictionary/'
        
        
                #file_path = tk.filedialog.askopenfilename()
        #img = Image.open(file_path)
        #image = ImageTk.PhotoImage(img)
        #####
        #path = 'csv_and_dict/dictionary/'
        class_dictionary_path = path + class_name.get() +'_dictionary.json'
        with open(class_dictionary_path) as f:
            data = json.load(f)
        class_name_pred_sigmoid = 'pred sigmoid ' + class_name.get()
        image_sorted_per_class = []
        for key, value in data.items():
            image_sorted_per_class.append((value['Unnamed: 0'],value[class_name_pred_sigmoid]))
        image_path = 'JPEGImages/'
        pagecount = math.ceil(len(image_sorted_per_class)/20)
        #image_dictionary =
        #image_list = list(image_dictionary)

        total_height = 0
        
        for i in range(20):
            try:
                #print(str(((currentpage-1)*20)+i+1))
                        
                img = Image.open(image_path + str(image_sorted_per_class[(currentpage-1)*20 + i][0]))
                image = ImageTk.PhotoImage(img)
                width,height_img = img.size
                total_height += height_img
                
                a = i*2
                panel_list[a].configure(image = image,height = height_img)
                panel_list[a].photo = image
                panel_list[a+1].configure(text = str((currentpage-1)*20 + i + 1) +') ' + class_name.get() +': ' + str(image_sorted_per_class[(currentpage-1)*20 + i][1]),height=2)        


            except:
                print("reached the end")
                break
                
        y_height = total_height + height_img*3 + 2*20 #height_image is for buffer
        print(y_height)
        self.canvas.config(scrollregion=(0,0,0,y_height))
        self.update()
        self.yscrollbar.config(command=self.canvas.yview)

        img_number.set(currentpage)
        pass
    
    def resetCurrentPage2(self):
        print('reset')
        global currentpage, pagecount
        currentpage = 1
        if self.button3["state"] != 'disabled':
            self.button3.config(state=tk.DISABLED)
        if self.button4["state"] != 'disabled':
            self.button4.config(state=tk.DISABLED)
        
        
if __name__ == "__main__":
    app = SampleApp()
    
    app.mainloop()