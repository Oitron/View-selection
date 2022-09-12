import os

import numpy as np


import torch
import torchvision
from torchvision import transforms

from scipy.special import comb


from sklearn.cluster import KMeans


from scipy.special import comb

import random


from tqdm.notebook import tqdm

from dataset import Dataset
from tools import get_img_feature, compute_FMscores_global, compute_FMscores_local

import json






# execute k-means cluster
# get predict labels
'''
    kpred: predict labels
'''
def k_means_cluster(features, nb_cluster):
    '''
        features: feature map
        nb_cluster: number of cluster
    '''
    k_means = KMeans(n_clusters=nb_cluster).fit(features)
    kpred = k_means.predict(features)
    return kpred



# get indices for whole set (all objects)
'''
    return: one clustering problem
'''
def one_tirage(set_length, nb_set, min_nb_set, max_nb_set, min_nb_sample, max_nb_sample):
    '''
        set_length: length of one class
        nb_set: number of class
        min_nb_set: minumum number of class need to tir 
        max_nb_set: maximum number of class need to tir
        min_nb_sample: minimum number of view need to tir in one class
        max_nb_sample: maximum number of view need to tir in one class
    '''
    if min_nb_set < 1 or max_nb_set > nb_set:
        print("min or max illegal, ", "min: ", min_nb_set, " max: ", max_nb_set)
        return np.array([])
    sample_sets = random_nb_sample(min_nb_set, max_nb_set)
    #print("sample sets: ", sample_sets)
    tir = np.sort(random.sample(range(nb_set), sample_sets))
    #print("tir: ", tir)
    indices = np.array([])
    for i in tir:
        ind_per_set = []
        nb_sample = random_nb_sample(min_nb_sample, max_nb_sample)
        #print("nb sample: ", nb_sample)
        for s in range(nb_sample):
            index = i*set_length+np.random.randint(set_length)
            while index in ind_per_set:
                index = i*set_length+np.random.randint(set_length)
            ind_per_set.append(index)
        indices = np.concatenate((indices, np.array(ind_per_set)), axis=None)
    return np.array(indices).flatten().astype(int)



# get random nomber of samples
def random_nb_sample(min, max):
    return random.randint(min, max)

# get random nomber of samples for different cluster
def random_nb_samples(min, max, nb_set):
    nb_samples = [random.randint(min, max) for p in range(nb_set)]
    return np.array(nb_samples)


def get_random_views(nb_sample, nb_set, set_length):
    indices = []
    for i in range(nb_set):
        tir = i*nb_set+np.sort(random.sample(range(set_length), nb_sample))
        indices+=tir.tolist()
    return np.array(indices)


'''
    return: ground truth and prediction for one clustering problem
'''
def cluster_pipeline(indices, dataset:Dataset, model, num_layer, transform, device):
    '''
        indices: image indices
        dataset: dataset where get images
        model: net(mobilenet)
        num_layer: index of layer for feature extraction
        transform: preprocess
        device: "cpu"/"cuda"
    '''
    features = []
    gt_labels = []
    for ind in indices:
        img, gt_label, _ = dataset.get_view(index=ind)
        gt_labels.append(gt_label)
        feature = get_img_feature(img=img,
                                  model=model,
                                  num_layer_ex=num_layer,
                                  transform=transform,
                                  device=device)
        #print(feature.shape)
        features.append(feature)
    gt_labels = np.array(gt_labels)
    #print("/---------- feature extraction done ---------/")
    # according to features use cluster to get prediction
    kpreds = k_means_cluster(features=features,
                             nb_cluster=len(np.unique(gt_labels)))
    #print("/---------- cluster process done ---------/")
    return gt_labels, kpreds


'''
    transfer indices to string(key of hashmap)
    ex: [1,5,10,8] -> "1,5,10,8"
'''
def indices_to_key(indices):
    key = ','.join([str(ind) for ind in indices])
    return key




##################################
#--------- main function --------#
##################################
def main():

    with open("config/compute_score.json") as json_file:
        variable_dict = json.load(json_file)



    DATA_FOLDER = variable_dict['data_set']['data_path']
    NB_CLASS = variable_dict['data_set']['num_classes']
    NB_OBJ = variable_dict['data_set']['num_objects']
    NB_VIEW = variable_dict['data_set']['num_views']

    LAST_MB_CONV_LAYER = 17

    NB_TIRAGE = variable_dict['nb_tirages']
    check_points = variable_dict['check_points']

    NB_SAMPLE_SET_MIN = variable_dict['nb_sample_set_min']
    NB_SAMPLE_SET_MAX = variable_dict['nb_sample_set_max']
    NB_SAMPLE_MIN = variable_dict['nb_sample_min']
    NB_SAMPLE_MAX = variable_dict['nb_sample_max']

    dataset = Dataset(
        root_folder=DATA_FOLDER,
        nb_class=NB_CLASS,
        nb_obj=NB_OBJ,
        nb_view=NB_VIEW,
    )


    print("data length: ", dataset.get_size())



    # use for feature extraction
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # verifier the number of tirage will not bigger than number of all combinations of clusting problems.
    all_combinations = 0

    for i in range(NB_SAMPLE_SET_MIN*NB_SAMPLE_MIN, (NB_SAMPLE_SET_MAX+1)*NB_SAMPLE_MIN, NB_SAMPLE_MIN):
        all_combinations += comb(NB_CLASS*NB_OBJ*NB_VIEW, i)

    if all_combinations > NB_TIRAGE:
        print(all_combinations, " > ", NB_TIRAGE)
        print("Number of tirage is smaller than number of all combinations, execute pipeline.")
    else:
        print(all_combinations, " < ", NB_TIRAGE)
        print("Number of tirage is bigger than number of all combinations, break.")
        return 

    #prepare cnn use for feature extraction
    mobileNet = torchvision.models.mobilenet_v2(pretrained=True)
    mobileNet.eval()


    # ----------------------- start tirage ---------------------- #
    SAVE_PATH = variable_dict['score_save_path']

    # use to store final scores, initialize to 0
    metrics_global = np.zeros(dataset.get_size())
    metrics_local = np.zeros(dataset.get_size())
    frequency = np.zeros(dataset.get_size())

    # used to store the combination of tirage
    dict = set()

    count = 0
    for i in tqdm(range(NB_TIRAGE)):
        #print("/**********----------------- tirage ", i+1, " ------------------*********/")
        indices = one_tirage(set_length=dataset.nb_view,
                             nb_set=dataset.nb_class*dataset.nb_obj,
                             min_nb_set=NB_SAMPLE_SET_MIN,
                             max_nb_set=NB_SAMPLE_SET_MAX,
                             min_nb_sample=NB_SAMPLE_MIN, 
                             max_nb_sample=NB_SAMPLE_MAX)
        key = indices_to_key(indices)
        while key in dict:
            indices = one_tirage(set_length=dataset.nb_view,
                                 nb_set=dataset.nb_class*dataset.nb_obj,
                                 min_nb_set=NB_SAMPLE_SET_MIN,
                                 max_nb_set=NB_SAMPLE_SET_MAX,
                                 min_nb_sample=NB_SAMPLE_MIN, 
                                 max_nb_sample=NB_SAMPLE_MAX)
            key = indices_to_key(indices)
        dict.add(key)
        #print("indices: ", indices)
        # according to result of tirage, get GT and features               
        gt_labels, kpreds = cluster_pipeline(indices=indices,
                                             dataset=dataset,
                                             model=mobileNet,
                                             num_layer=LAST_MB_CONV_LAYER,
                                             transform=preprocess,
                                             device=DEVICE)
        #print("labels: ", gt_labels)
        #print("predicts: ", kpreds)
        # according to prediction, compute FMI global and local
        FMI_global = compute_FMscores_global(gt_labels, kpreds)
        FMI_locals = compute_FMscores_local(gt_labels, kpreds)
        #print("FMI global: ", FMI_global)
        #print("FMI locals: ", FMI_locals)
        metrics_global[indices] += FMI_global
        metrics_local[indices] += FMI_locals
        frequency[indices] += 1
        #print("/---------- metrics compute done ---------/")
        # set check points
        if i+1 == check_points[count]:
            each_save_path = os.path.join(SAVE_PATH, "tir_"+str(check_points[count]))
            if not os.path.exists(each_save_path):
                os.mkdir(each_save_path)
            np.save(os.path.join(each_save_path, "frequency_tir_"+str(check_points[count])+".npy"), np.array(frequency))
            np.save(os.path.join(each_save_path, "metrics_global_tir_"+str(check_points[count])+".npy"), np.array(metrics_global))
            np.save(os.path.join(each_save_path, "metrics_local_tir_"+str(check_points[count])+".npy"), np.array(metrics_local))
            print(check_points[count], " tirage successfully saved!")
            count+=1




if __name__ == '__main__':
    main()
        