#!/usr/bin/env python
# coding: utf-8

# In[19]:


from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os

from tqdm import tqdm
import numpy as np
import torch
import os

        



# In[38]:



per_class = 20

aux_num = 370

val_catgs = {}

# In[38]:



class FlowDataset_Train(data.Dataset):
    def __init__(self,mode, transform=None, target_transform=None):
        super(FlowDataset_Train, self).__init__()
        self.root = '..' + os.sep + mode
        self.transform = transform
        self.target_transform = target_transform
        self.aux_only = []
        self.child_only = []
        self.aux_labels = []

        #load subfloders
        classes = os.listdir(self.root)

        if not mode == 'train':
        	for ind,clss in enumerate(classes):
        		print(str(ind)+': '+clss)
        		val_catgs[ind] = clss
        #load image from subfloders
        self.data = []

        self.label = []

        tmp_data = []
        tmp_label = []


        
        for class_name in classes:
            root_path = os.path.join(self.root,class_name)
            sub_data_name = os.listdir(root_path)

            tmp_data = []
            tmp_label = []

            idx = np.arange(len(sub_data_name))
            np.random.shuffle(idx)

            
            for i,item_index in enumerate(idx):
                file_path = os.path.join(root_path,sub_data_name[item_index])


                
                #打开图片
                im = Image.open(file_path)
                im = np.array(im)
                
                tmp_data.append(im/255.0)
                tmp_label.append(classes.index(class_name)) 


                if mode == 'train' and i >=  per_class-1:
                    break

            self.data.extend(tmp_data)
            self.label.extend(tmp_label)
            self.child_only.extend(tmp_label)

        self.aux_labels.append(self.child_only)

        if mode =='train':

            # load aux data 
            print('Aux Data Loading.....')
            tmp_data = []
            tmp_label = []

            classes = os.listdir('../backup_train')
            idx_cls = np.arange(len(classes))
            np.random.shuffle(idx_cls)


            for i,class_name_index in enumerate(idx_cls):
                class_name = classes[class_name_index]

                if i>= aux_num:
                    break
                root_path = os.path.join('../backup_train',class_name)
                sub_data_name = os.listdir(root_path)

                tmp_data = []
                tmp_label = []

                idx = np.arange(len(sub_data_name))
                np.random.shuffle(idx)




                
                for j,item_index in enumerate(idx):



                    file_path = os.path.join(root_path,sub_data_name[item_index])


                    
                    #打开图片
                    im = Image.open(file_path)
                    im = np.array(im)
                    
                    tmp_data.append(im/255.0)
                    tmp_label.append(i+42) 


                    # if j >=  per_class-1:
                    #     break

                self.data.extend(tmp_data)
                self.label.extend(tmp_label)
                self.aux_only.extend(tmp_label)


            print('Aux Data Finish.....')
            self.aux_labels.append(self.aux_only)
            print(len(self.aux_labels))

            _, num = np.unique(np.array(self.label), return_counts=True)
            print(len(num))
            _, num = np.unique(np.array(self.aux_only), return_counts=True)
            print(len(num))

        
            # if mode == 'train':
                

            #     idx = np.arange(len(tmp_data))
            #     np.random.shuffle(idx)

            #     # self.data.extend([tmp_data[item] for item in idx[:per_class]])
            #     # self.label.extend([tmp_label[item] for item in idx[:per_class]])

            #     self.data.extend([tmp_data[item] for item in idx])
            #     self.label.extend([tmp_label[item] for item in idx])
            # else:
            #     self.data.extend(tmp_data)
            #     self.label.extend(tmp_label)
                
    def __getitem__(self, idx):
        # print(idx,len(self.data))

        x = self.data[idx]

        x = x.reshape((1,28,28))
        if self.transform:
            x = self.transform(x)
            x = torch.from_numpy(x).float()
            
        return torch.from_numpy(x), self.label[idx]

    def __len__(self):
        return len(self.data)

