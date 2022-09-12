import numpy as np
import os

from tools import img_load


'''
    base dataset
'''
class Dataset:
    def __init__(self, root_folder, nb_class, nb_obj, nb_view):
        '''
            root_folder: path of dataset
            nb_class: number of class
            nb_obj: number of object per class
            nb_view: number of view per object
        '''
        self.root_folder = root_folder
        self.nb_class = nb_class
        self.nb_obj = nb_obj
        self.nb_view = nb_view
        self.classes = sorted(os.listdir(root_folder))

    '''
    return: image(np array), label and path of image
    '''
    def get_view(self, index):
        '''
            index: index of image in dataset
        '''
        if index >= self.nb_class*self.nb_obj*self.nb_view:
            print("index out of range!")
            return np.array([])
        label = index//(self.nb_obj*self.nb_view)
        o = (index-label*self.nb_obj*self.nb_view)//self.nb_view
        i = index-label*self.nb_obj*self.nb_view - o*self.nb_view
        objs = sorted(os.listdir(os.path.join(self.root_folder, self.classes[label])))
        views = os.listdir(os.path.join(self.root_folder, self.classes[label], objs[o]))
        #views.sort(key=lambda x:int(x.split('\\')[-1].split('.')[0]))
        img_path = os.path.join(self.root_folder, self.classes[label], objs[o], views[i])
        img = img_load(img_path)
        return img, label, img_path

    '''
        return size of dataset(view total)
    '''
    def get_size(self):
        return self.nb_class*self.nb_obj*self.nb_view

    '''
        return classes(labels) names
    '''
    def get_labels(self):
        return self.classes
    


    

    


