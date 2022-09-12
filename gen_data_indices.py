import heapq
import numpy as np
from dataset import Dataset
import os
import shutil
import cv2
import random


def get_max_elements(data, nb_max=1):
    data = data.tolist()
    max_element = heapq.nlargest(nb_max, data)
    max_index = []
    for elem in max_element:
        index = data.index(elem)
        max_index.append(index)
        data[index] = float("inf")
    return np.array(max_element), np.array(max_index)

def get_top_train_indices(scores, nb_train_view, ascending=True):
    nb_class, nb_obj, nb_view = scores.shape
    if ascending:
        scores = -scores
    indices = np.argsort(scores, axis=2)
    scores = abs(np.sort(scores, axis=2))
    base = np.arange(0, nb_view*nb_class*nb_obj, nb_view)
    base = base.repeat(nb_view)
    base = base.reshape(nb_class, nb_obj, nb_view)
    indices = indices + base

    scores = scores.transpose(0,2,1)
    indices = indices.transpose(0,2,1)

    train_indices = np.zeros((nb_class, nb_train_view))
    a = int(nb_train_view/nb_obj)
    b = nb_train_view%nb_obj
    for i in range(nb_class):
        train_indices[i,0:a*nb_obj] = indices[i,0:a,:].flatten()
        if b!=0:
            train_indices[i,a*nb_obj:a*nb_obj+b] = indices[i,a,get_max_elements(scores[i,a,:], nb_max=b)[1]]
    train_indices = train_indices.flatten()
    return train_indices.astype(np.int64)

def get_rand_train_indices(scores, nb_train_view):
    nb_class, nb_obj, nb_view = scores.shape
    nb_set = nb_class
    set_length = nb_obj * nb_view
    ex_indices = get_top_train_indices(scores, nb_train_view)
    indices = np.arange(nb_set*set_length)
    indices = np.setdiff1d(indices, ex_indices, False) # exclue top views
    train_indices = []
    inter = set_length - nb_train_view
    for i in range(nb_set):
        train_indices.append(random.sample(list(indices[i*inter:(i+1)*inter]), 
                                        int(nb_train_view)))
    train_indices = np.array(train_indices).flatten()
    return train_indices.astype(np.int64)

def get_train_indices(scores, nb_train_view, type):
    if type == "top":
        return get_top_train_indices(scores, nb_train_view, ascending=True)
    elif type == "random":
        return get_rand_train_indices(scores, nb_train_view)
    else:
        return np.empty()



def get_test_indices(scores, train_ind, cross_view):
    nb_class, nb_obj, nb_view = scores.shape
    total_indices = np.arange(nb_class*nb_obj*nb_view)
    if cross_view == "union":
        return total_indices.astype(np.int64)
    train_indices = np.reshape(train_ind, (nb_class, -1))
    #print("train indices: ")
    #print(train_indices)
    ex_indices = []
    for c in range(nb_class):
        ind = np.unique(train_indices[c]%nb_view+nb_view*c*nb_obj)
        #print(train_indices[c]%nb_view+nb_view*c*nb_obj)
        #print("unique indices: ")
        #print(ind)
        ex_ind = []
        for o in range(nb_obj):
            ex_ind.extend(ind+nb_view*o)
        ex_indices.extend(ex_ind)
    ex_indices = np.array(ex_indices)
    #print("ex_indices: ")
    #print(ex_indices)
    if cross_view == "inter":
        return ex_indices.astype(np.int64)
    elif cross_view == "complet":
        test_indices = np.setdiff1d(total_indices, ex_indices)
        return test_indices.astype(np.int64)











def save_views_and_labels(data_save_folder, indices, base_set:Dataset):
    if os.path.exists(data_save_folder):
        shutil.rmtree(data_save_folder)
    img_folder = os.path.join(data_save_folder, "imgs")
    label_folder = os.path.join(data_save_folder, "labels")
    os.makedirs(img_folder)
    os.makedirs(label_folder)
    for i in indices:
        img, label, path = base_set.get_view(index=i)
        path_str = path.split('\\')
        img_file_name = path_str[-3]+'-'+path_str[-2]+'-'+path_str[-1].replace(".jpg",'.png')
        label_file_name = img_file_name.replace(".png",'.txt')
        img_save_path = os.path.join(img_folder, img_file_name)
        label_save_path = os.path.join(label_folder, label_file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_save_path, img)
        label_file = open(label_save_path, 'w')
        label_file.write(str(label))






'''
parser = argparse.ArgumentParser()
parser.add_argument('--data-save-folder', type=str, required=True)
parser.add_argument('--nb-view', type=int, required=True)




def main():

    global args 
    args = parser.parse_args()


    ROOT_FOLDER = "score_pipeline/DATA/classifier/fondGris - 003030/train"
    NB_CLASS = 7
    NB_OBJ = 1
    NB_VIEW = 48

    LABEL_NAMES = np.array([
        "conveyor",
        "fruit crate",
        "pallet",
        "cardboard box",
        "plastic crate",
        "robot arm",
        "wooden box",
    ])
    
    CAM_DEGREES = np.array([
        [0 ,   0],
        [30,   0],
        [30,  90],
        [60,  90],
        [90,  90],
        [30, 120],
        [60, 120],
        [90, 120],
        [30, 150],
        [60, 150],
        [90, 150],
        [30, 180],
        [60,   0],
        [60, 180],
        [90, 180],
        [30, 210],
        [60, 210],
        [90, 210],
        [30, 240],
        [60, 240],
        [90, 240],
        [30, 270],
        [60, 270],
        [90,   0],
        [90, 270],
        [30, 300],
        [60, 300],
        [90, 300],
        [30, 330],
        [60, 330],
        [90, 330],
        [30,  30],
        [60,  30],
        [90,  30],
        [30,  60],
        [60,  60],
        [90,  60],
    ])



    base_set = Dataset(
        root_folder=ROOT_FOLDER,
        nb_class=NB_CLASS,
        nb_obj=NB_OBJ,
        nb_view=NB_VIEW,
        cam_degrees=None,
    )

    scores = np.load("score_pipeline/score_s_tir_30000.npy")
    scores = scores.reshape((NB_CLASS, NB_OBJ, NB_VIEW))
    
    top_train_indices = get_train_indices(scores, args.nb_view, ascending=True).flatten()
    bottom_train_indices = get_train_indices(scores, args.nb_view, ascending=False).flatten()
    
    #train_indices = np.concatenate((top_train_indices, bottom_train_indices))

    save_views_and_labels(
        data_save_folder=args.data_save_folder,
        indices=bottom_train_indices,
        base_set=base_set,
    )






if __name__ == '__main__':
    main()
    print("save success!")
'''