import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score 
from sklearn.metrics.cluster import fowlkes_mallows_score

#---load image as narray from file---#
def img_load(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img_save(savepath, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(savepath, img)

#---normalize img pixel val to [0, 1]---#
def img_normalize(img):
    val_range = img.max() - img.min()
    return (img - img.min()) / val_range

#---load label from file---#
def label_load(filepath):
    f = open(filepath)
    line = f.readline()
    label = line
    while line:
        line = f.readline()
        if line:
            label += "/"
        label += line
    return label


# calculate FMI global
'''
    return: FMI global
'''
def compute_FMscores_global(lab1, lab2):
    '''
        lab1: grand truth
        lab2: predict labels
    '''
    return fowlkes_mallows_score(lab1, lab2)

# calculer FMI local/individual 
def compute_cocluster_mat(labels):
    mat = np.zeros([labels.shape[0], labels.shape[0]])
    for i in range(labels.shape[0]):
        for j in range(i, labels.shape[0]):
            mat[i, j] = int(labels[i] == labels[j])
            mat[j, i] = mat[i, j]
    return mat
'''
    FMI_arr: FMI local
'''
def compute_FMscores_local(lab1, lab2):
    '''
        lab1: grand truth
        lab2: predict labels
    '''
    mat1 = compute_cocluster_mat(lab1)
    mat2 = compute_cocluster_mat(lab2)
    
    FP_arr = np.sum((mat1 - mat2 == -1), axis = 1)
    FN_arr = np.sum((mat1 - mat2 == 1), axis = 1)
    TP_arr = np.sum((mat1 + mat2 == 2), axis = 1) - 1

    FMI_arr = np.zeros(TP_arr.shape)
    for i in range(len(TP_arr)):
        if TP_arr[i] != 0:
            FMI_arr[i] = float(TP_arr[i] / float(np.sqrt((TP_arr[i] + FP_arr[i]) * (TP_arr[i] + FN_arr[i]))))
        else:
            FMI_arr[i] = 0

    return FMI_arr



#---compute global F1 score---#
def compute_f1_global(gt_labels, predicts):
    return f1_score(gt_labels, predicts, average='macro')
    
#---compute local F1 score---#
def compute_f1_local(gt_labels, predicts):
    f1_s = []
    label_set = np.unique(gt_labels)
    for label in label_set:
        label_indices = np.array(np.where(gt_labels == label)).flatten()
        label_per_class = gt_labels[label_indices]
        predict_per_class = predicts[label_indices]
        #print(label_per_class.shape)
        #print(label_per_class)
        #print(predict_per_class.shape)
        #print(predict_per_class)
        f1 = f1_score(label_per_class, predict_per_class, average='micro')
        f1_s.append(f1)
    return f1_s


def all_metrics_test(ground_truth, predict, nb_class, average='macro'):
    cm = confusion_matrix(ground_truth, predict)
    f1 = f1_score(ground_truth, predict, labels=np.arange(nb_class), average=average)
    ps = precision_score(ground_truth, predict, labels=np.arange(nb_class), average=average)
    rs = recall_score(ground_truth, predict, labels=np.arange(nb_class), average=average)
    return f1, cm, ps, rs





# feature extraction
# extract features for an image
'''
    feature_maps: flattened (feature vector) 
'''
def get_img_feature(img, model, nb_layer_fix, num_layer_ex, transform, device="cuda"):
    '''
        img: image(np array)
        model: net(mobilenet)
        num_layer: index of layer for feature extraction
        transform: preprocess
        device: "cpu"/"cuda"
    '''
    model.to(device)
    img_batch = transform(img).unsqueeze(0)
    input = img_batch.to(device)
    for i in range(num_layer_ex):
        if i < nb_layer_fix:
            layer = model.features[i]
        else:
            layer = model.train_features[i-nb_layer_fix]
        output = layer(input)
        input = output
    output = input
    feature_maps = output.squeeze(0).cpu().detach()
    return feature_maps.numpy().flatten()