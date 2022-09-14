import os

import shutil
import numpy as np

from dataset import Dataset

import torch
from torchvision import transforms
from models import MobileNet

from tools import compute_f1_local, img_load

from sklearn.neighbors import KNeighborsClassifier as Knn


from add_noise import add_saltPepper_noise, add_gaussian_noise

from tools import get_img_feature, all_metrics_test
from gen_data_indices import get_train_indices, get_test_indices


import warnings 
import json





def compute_feature_set(data, model, nb_layer_fix, num_layer_ex, transform, device):
    feature_set = []
    for i in range(len(data)):
        path = data[i,0]
        img = img_load(path)
        feature = get_img_feature(img=img, model=model, nb_layer_fix=nb_layer_fix, num_layer_ex=num_layer_ex, transform=transform, device=device)
        feature_set.append(feature)
    feature_set = np.array(feature_set)
    ground_truth = data[:,1]
    return feature_set, ground_truth


def test_pipeline(
    train_data, test_data, train_size_pc, nb_class,
    model, nb_layer_fix, layer_ex, transform, device
    ):

    train_features, train_gt_labels = compute_feature_set(train_data, model, nb_layer_fix, layer_ex, transform, device)
    test_features, test_gt_labels = compute_feature_set(test_data, model, nb_layer_fix, layer_ex, transform, device)

    f1_max = 0
    n_max = 1
    cm = []
    pre_max = []
    f1_l = []
    for n in range(1, train_size_pc+1):
        knn = Knn(n_neighbors=n)
        knn.fit(train_features, train_gt_labels)
        pre = knn.predict(test_features)
        results = all_metrics_test(test_gt_labels, pre, nb_class, average='macro')
        f1 = results[0]
        if f1 > f1_max:
            f1_max = f1
            cm = results[1]
            pre_max = pre
            f1_l = compute_f1_local(test_gt_labels, pre)
            n_max = n

    print("k_max for knn: ", n_max)

    return f1_max, cm, pre_max
        
    





##################################
#--------- main function --------#
##################################
def main():


    with open("config/train_test_knn.json") as json_file:
        variable_dict = json.load(json_file)


    TRAIN_DATA_FOLDER = variable_dict['train_set']['data_path']

    TEST_DATA_FOLDER = variable_dict['test_set']['data_path']

    TRAIN_NB_OBJ = variable_dict['train_set']['num_objects']
    TRAIN_NB_VIEW = variable_dict['train_set']['num_views']
    TRAIN_NB_CLASS = variable_dict['train_set']['num_classes']

    TEST_NB_OBJ = variable_dict['test_set']['num_objects']
    TEST_NB_VIEW = variable_dict['test_set']['num_views']
    TEST_NB_CLASS = variable_dict['test_set']['num_classes']

    score_file = variable_dict['score_file']
    num_execution = variable_dict['num_execution']
    cross_view = variable_dict['cross_view']
    sub_train_size = variable_dict['train_size_per_class']
    sub_test_size = variable_dict['test_size_per_class']
    train_type = variable_dict['training_type']


    weights_files_dir = variable_dict['weights_files_dir']
    nb_layer_fix = variable_dict['nb_layer_fix']
    

    if train_type == "top":
        num_execution = 1
   
   

    train_datasets = []
    for i in range(len(TRAIN_DATA_FOLDER)):
        train_datasets.append(
            Dataset(
                root_folder=TRAIN_DATA_FOLDER[i],
                nb_class=TRAIN_NB_CLASS,
                nb_obj=TRAIN_NB_OBJ,
                nb_view=TRAIN_NB_VIEW,
            )
        )

    test_datasets = []
    for i in range(len(TEST_DATA_FOLDER)):
        test_datasets.append(
            Dataset(
                root_folder=TEST_DATA_FOLDER[i],
                nb_class=TEST_NB_CLASS,
                nb_obj=TEST_NB_OBJ,
                nb_view=TEST_NB_VIEW,
            )
        )


    LAST_MB_CONV_LAYER = 17


    # use for feature extraction
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        #add_saltPepper_noise(0.02, 0.2),
        #add_gaussian_noise(0.0, 1.0, 10, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # load net for feature extration
    model = MobileNet(num_classes=TRAIN_NB_CLASS).to(DEVICE)
    if weights_files_dir != "None":
        model = MobileNet(num_classes=TRAIN_NB_CLASS, nb_layer_fix=nb_layer_fix).to(DEVICE)
        weights_dir = os.path.join(weights_files_dir, "models")
        weights_files = sorted(os.listdir(weights_dir))

    scores = np.load(score_file)
    #print("scores: ")
    #print(scores.shape)
    #print(scores)
    #print(metrics_local_20000.shape)
    scores = scores.reshape((TRAIN_NB_CLASS, TRAIN_NB_OBJ, TRAIN_NB_VIEW))
    

    f1_score = []
    confusion_matrix = []
    test_img_paths = []
    test_gt_labels = []
    test_pre_labels = []
    for round in range(num_execution):
        print("************----------------{}/{} execution ----------------************".format(round+1, num_execution))
        
          
        train_indices = []
        test_indices = []
        if cross_view != "other":
            train_indices = get_train_indices(scores, sub_train_size, type=train_type)
            test_indices = get_test_indices(scores, train_indices, cross_view)
        else:
            if train_type == "top":
                train_indices = get_train_indices(scores, sub_train_size, type=train_type)
                ex_indices = get_train_indices(scores, sub_test_size, type=train_type)
                test_indices = get_test_indices(scores, ex_indices, "inter")
            elif train_type == "random":
                if sub_train_size >= sub_test_size:
                    train_indices = get_train_indices(scores, sub_train_size, type=train_type)
                    ex_indices = train_indices.reshape((TRAIN_NB_CLASS, sub_train_size))
                    test_indices = ex_indices[:,0:sub_test_size].flatten()
                else:
                    test_indices = get_train_indices(scores, sub_test_size, type=train_type)
                    ex_indices = test_indices.reshape((TRAIN_NB_CLASS, sub_test_size))
                    train_indices = ex_indices[:,0:sub_train_size].flatten()

        train_indices = sorted(train_indices)
        test_indices = sorted(test_indices)

        print("Train dataset size: ", len(train_indices))
        print(train_indices)
        print("Eval dataset size: ", len(test_indices))
        print(test_indices)


        train_data = []
        for t in range(len(TRAIN_DATA_FOLDER)):
            for i in train_indices:
                _, label, img_path = train_datasets[t].get_view(i)
                train_data.append([img_path, label])
        train_data = np.array(train_data)

        test_data = []
        for t in range(len(TEST_DATA_FOLDER)):
            for i in test_indices:
                _, label, img_path = test_datasets[t].get_view(i)
                test_data.append([img_path, label])
        test_data = np.array(test_data)

        test_img_paths.append(test_data[:,0])
        test_gt_labels.append(test_data[:,1])

        #print(train_data)
        #print()
        #print(test_data)

        #load model
        if weights_files_dir != "None":
            state_dict = model.state_dict()
            for n,p in torch.load(os.path.join(weights_dir, weights_files[round]), map_location=lambda storage, loc: storage).items():
                if n in state_dict.keys():
                    state_dict[n].copy_(p)
                else:
                    raise KeyError(n)
        model.eval()

        #print(model)

        f1, cm, pre = test_pipeline(
            train_data, test_data, sub_train_size, nb_class=TRAIN_NB_CLASS,
            model=model, nb_layer_fix=nb_layer_fix, layer_ex=LAST_MB_CONV_LAYER, transform=preprocess, device=DEVICE,
        )

        f1_score.append(f1)
        confusion_matrix.append(cm)
        test_pre_labels.append(pre)

        print("confusion matrix: ")
        print(cm)




    f1_mean = np.array(f1_score).mean()
    max_round = np.array(f1_score).argmax()

    print("F1 score: {:.3f}".format(f1_mean))


    #path = os.path.join("res/knn", train_type, TEST_DATA_FOLDER[0].replace('/','_'))
    #if os.path.exists(path):
    #    shutil.rmtree(path)
    #os.makedirs(path)
    #np.save(os.path.join(path, "confusion_matrix.npy"), confusion_matrix[max_round])
    #np.save(os.path.join(path, "test_img_paths.npy"), test_img_paths[max_round])
    #np.save(os.path.join(path, "test_gt_labels.npy"), test_gt_labels[max_round])
    #np.save(os.path.join(path, "test_pre_labels.npy"), test_pre_labels[max_round])



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()



