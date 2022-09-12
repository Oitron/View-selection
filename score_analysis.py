import os
import numpy as np
import matplotlib.pyplot as plt
import heapq
from dataset import Dataset


import json



# get first nb_max elements for a set
def get_max_elements(data, nb_max=1):
    data = data.tolist()
    max_element = heapq.nlargest(nb_max, data)
    max_index = []
    for elem in max_element:
        index = data.index(elem)
        max_index.append(index)
        data[index] = float("inf")
    return np.array(max_element), np.array(max_index)

def get_min_elements(data, nb_min=1):
    data = data.tolist()
    min_element = heapq.nsmallest(nb_min, data)
    min_index = []
    for elem in min_element:
        index = data.index(elem)
        min_index.append(index)
        data[index] = float("inf")
    return np.array(min_element), np.array(min_index)


def get_max_views(metrics, nb_set, set_length, nb_max=1):
    if nb_set*set_length != len(metrics):
        print("metrics formet error, shape not right: ", len(metrics), " != ", nb_set*set_length)
        return np.array([])
    max_views = []
    for i in range(nb_set):
        m = metrics[i*set_length : (i+1)*set_length]
        maxs, maxs_indices = get_max_elements(m, nb_max=nb_max)
        maxs_indices += i*set_length
        res = np.array((maxs, maxs_indices.astype(int))).transpose((1,0))
        max_views.append(res)
    return np.array(max_views)


def get_min_views(metrics, nb_set, set_length, nb_min=1):
    if nb_set*set_length != len(metrics):
        print("metrics formet error, shape not right: ", len(metrics), " != ", nb_set*set_length)
        return np.array([])
    min_views = []
    for i in range(nb_set):
        m = metrics[i*set_length : (i+1)*set_length]
        #print("data: ", m)
        mins, mins_indices = get_min_elements(m, nb_min=nb_min)
        mins_indices += i*set_length
        #print("min indices: ", mins_indices)
        res = np.array((mins, mins_indices.astype(int))).transpose((1,0))
        min_views.append(res)
    return np.array(min_views)


def display_views(metrics_views, dataset:Dataset, nb_disp_view, save_path):
    fig,axes = plt.subplots(nb_disp_view, 
                            metrics_views.shape[0], 
                            figsize=(4*metrics_views.shape[0], 4*nb_disp_view))
    for i in range(metrics_views.shape[0]*nb_disp_view):
        r = i//nb_disp_view
        c = i-r*nb_disp_view
        index = int(metrics_views[r,c,1])
        val = np.round(metrics_views[r,c,0], 3)
        img, _, path = dataset.get_view(index=index)
        path_str = path.split('\\')
        title1 = path_str[-3]+'-'+path_str[-2]+'-'+path_str[-1].replace(".jpg",'')
        #title2 = " ("+str(degrees[index%dataset.nb_view,0])+", "+str(degrees[index%dataset.nb_view,1])+")"
        title2 = ""
        axes[c,r].set_title(title1 + title2)
        axes[c,r].text(0, 50, "score: "+str(val), size=10, bbox=dict(boxstyle="square",ec=(1.0, 0.5, 0.5),fc=(1.0, 0.8, 0.8),))
        axes[c,r].axis('off')
        axes[c,r].imshow(img)
    plt.savefig(save_path)



def main():

    with open("config/compute_score.json") as json_file:
        variable_dict = json.load(json_file)


    SCORE_FOLDER = variable_dict['score_save_path']
    check_points = variable_dict['check_points']
    FOLDERS = os.listdir(SCORE_FOLDER)

    DATA_FOLDER = variable_dict['data_set']['data_path']
    NB_OBJ = variable_dict['data_set']['num_objects']
    NB_VIEW = variable_dict['data_set']['num_views']
    NB_CLASS = variable_dict['data_set']['num_classes']

    SAVE_DIR = variable_dict['score_analysis_results_save_path']


    dataset = Dataset(
        root_folder=DATA_FOLDER,
        nb_class=NB_CLASS,
        nb_obj=NB_OBJ,
        nb_view=NB_VIEW,
    )

    check_folders = []
    for folder in FOLDERS:
        if int(folder.split('_')[-1].split('.')[0]) in check_points:
            check_folders.append(folder)
    check_folders.sort(key=lambda x:int(x.split('_')[-1]))



    res_all = []

    for folder in check_folders:
        files = os.listdir(os.path.join(SCORE_FOLDER, folder))
        print(files)
        res = []
        for file in files:
            data = np.load(os.path.join(SCORE_FOLDER, folder, file))
            res.append(data)
        res_all.append(res)

    res_all = np.array(res_all)
    print(res_all.shape)

    metrics_global = res_all[:,1,:] / res_all[:,0,:]
    metrics_local = res_all[:,2,:] / res_all[:,0,:]
    print(metrics_global.shape)
    print(metrics_local.shape)


    ##### ----------- draw score curves ---------- #####
    color=['b','c','m','y']
    marker=['o','v','s','D']

    fig,ax = plt.subplots(figsize=(12, 10))

    line1 = ax.plot(check_points, np.max(metrics_local, axis=1), 
                    label="max", linewidth=1, marker=marker[0], color=color[0])
    line2 = ax.plot(check_points, np.min(metrics_local, axis=1), 
                    label="min", linewidth=1, marker=marker[1], color=color[1])
    line3 = ax.plot(check_points, np.mean(metrics_local, axis=1), 
                    label="mean", linewidth=1, marker=marker[2], color=color[2])
    line4 = ax.plot(check_points, np.std(metrics_local, axis=1), 
                    label="std", linewidth=1, marker=marker[3], color=color[3])


    #plt.ylim((0, 1))
    ax.set_xlabel("Number of CP", size=15)
    ax.set_ylabel("score (SS)", size=15)
    #ax.set_title("maximum/random views")
    ax.legend(prop={'size':15})
    plt.tick_params(labelsize=13)
    plt.savefig(os.path.join(SAVE_DIR, "scores_var_tirages.png"))
    print("score curves saved")


    ##### ---------- chose fit score ----------- #####
    nb_tirages = input("Enter fit nb tirages: ")
    nb_tirages = int(nb_tirages)
    ind = check_points.index(nb_tirages)
    scores_fit = metrics_local[ind,:]
    print(scores_fit.shape)
    np.save(os.path.join(SAVE_DIR, "score_s_tir_fit.npy"), scores_fit)

    NB_DISP_VIEW = 5

    MAX_FIG_PATH = os.path.join(SAVE_DIR, "ex_max_views.png")
    MIN_FIG_PATH = os.path.join(SAVE_DIR, "ex_min_views.png")

    metrics_local_max_views = get_max_views(
        metrics=scores_fit, 
        nb_set=NB_CLASS,
        set_length=NB_OBJ*NB_VIEW,
        nb_max=NB_DISP_VIEW
    )

    #print(metrics_local_max_views.shape)
    #print(metrics_local_max_views)

    display_views(
        metrics_views=metrics_local_max_views, 
        dataset=dataset, 
        nb_disp_view=NB_DISP_VIEW,
        save_path=MAX_FIG_PATH
    )

    metrics_local_min_views = get_min_views(
        metrics=scores_fit, 
        nb_set=NB_CLASS,
        set_length=NB_OBJ*NB_VIEW,
        nb_min=NB_DISP_VIEW
    )

    #print(metrics_local_min_views.shape)
    #print(metrics_local_min_views)

    display_views(
        metrics_views=metrics_local_min_views, 
        dataset=dataset, 
        nb_disp_view=NB_DISP_VIEW,
        save_path=MIN_FIG_PATH
    )



if __name__ == '__main__':
    main()
        