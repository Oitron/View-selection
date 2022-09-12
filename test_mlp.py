import time
import shutil
import numpy as np
import os
import torch
from gen_data_indices import get_test_indices, get_train_indices
from models import MobileNet, SqueezeNet, MnasNet, DenseNet, ResNet, InceptionV3
from tools import img_normalize, compute_f1_global, compute_f1_local
from load_data import IVData, IVData_F, UnityData, IVData_D

from torch.utils.data import DataLoader
from torchvision import transforms
from add_noise import add_saltPepper_noise, add_gaussian_noise

import matplotlib.pyplot as plt
import sklearn.metrics

from dataset import Dataset
import json



MOBILE_NET   = "MOBILENET"
SQUEEZE_NET  = "SQUEEZENET"
MNAS_NET     = "MNASNET"
DENSE_NET    = "DENSENET"
RES_NET      = "RESNET"
INCEPTION_V3 = "INCEPTIONV3"




def main():

    with open("config/test_mlp.json") as json_file:
        variable_dict = json.load(json_file)

    TEST_DATA_FOLDER = variable_dict['test_set']['data_path']

    TEST_NB_OBJ = variable_dict['test_set']['num_objects']
    TEST_NB_VIEW = variable_dict['test_set']['num_views']
    TEST_NB_CLASS = variable_dict['test_set']['num_classes']

    cross_view = variable_dict['cross_view']
    sub_test_size = variable_dict['test_size_per_class']

    score_file = variable_dict['score_file']

    model_name = variable_dict['model'].upper()
    weights_files_dir = variable_dict['weights_files_dir']
    nb_layer_fix = variable_dict['nb_layer_fix']
    batch_size = variable_dict['batch_size']

    test_type = weights_files_dir.split('/')[-1]

    device_set = variable_dict['device']

    

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu" and device_set == "cuda":
        print("No GPU support.")
    else:
        DEVICE = device_set

    print("Using device: ", DEVICE)




    test_img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        #add_saltPepper_noise(0.05, 0.7),
        #add_gaussian_noise(0.0, 1.0, 10, 0.7),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if model_name == INCEPTION_V3:
        test_img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            #add_saltPepper_noise(0.05, 0.7),
            #add_gaussian_noise(0.0, 1.0, 10, 0.7),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    model_names = [
        MOBILE_NET,
        SQUEEZE_NET,  
        MNAS_NET,
        DENSE_NET,
        RES_NET,
        INCEPTION_V3,
    ]

    if model_name not in model_names:
        print("model name not correct: ")
    else: 
        print("Using model: ", model_name)


    if model_name == MOBILE_NET:
        model = MobileNet(num_classes=TEST_NB_CLASS, nb_layer_fix=nb_layer_fix).to(DEVICE)
    elif model_name == SQUEEZE_NET:
        model = SqueezeNet(num_classes=TEST_NB_CLASS).to(DEVICE)
    elif model_name == MNAS_NET:
        model = MnasNet(num_classes=TEST_NB_CLASS).to(DEVICE)
    elif model_name == DENSE_NET:
        model = DenseNet(num_classes=TEST_NB_CLASS).to(DEVICE)
    elif model_name == RES_NET:
        model = ResNet(num_classes=TEST_NB_CLASS).to(DEVICE)
    elif model_name == INCEPTION_V3:
        model = InceptionV3(num_classes=TEST_NB_CLASS).to(DEVICE)

    weights_dir = os.path.join(weights_files_dir, "models")
    indices_dir = os.path.join(weights_files_dir, "train_indices")
    
    weights_files = sorted(os.listdir(weights_dir))
    train_indices = sorted(os.listdir(indices_dir))
    n_test = len(weights_files)

    #load data
    test_set = []
    for l in range(len(TEST_DATA_FOLDER)):
        test_set.append(
            Dataset(
                root_folder=TEST_DATA_FOLDER[l],
                nb_class=TEST_NB_CLASS,
                nb_obj=TEST_NB_OBJ,
                nb_view=TEST_NB_VIEW,
            )
        )

    
    print("Number of test: ", n_test)

    scores = np.load(score_file)
    scores = scores.reshape((TEST_NB_CLASS, TEST_NB_OBJ, TEST_NB_VIEW))

    visualized = False
    verbose = 5

    load_time = 0
    test_time = 0



    f1_score = []
    confusion_matrix = []
    test_img_paths = []
    test_gt_labels = []
    test_pre_labels = []

    for n in range(n_test):
        print("******---------- {}/{} test ----------******".format(n+1, n_test))

        train_ind = np.load(os.path.join(indices_dir, train_indices[n]))
        sub_train_size = int(len(train_ind)/TEST_NB_CLASS)
        test_indices = []
        if cross_view != "other":
            test_indices = get_test_indices(scores, train_ind, cross_view)
        else:
            if test_type == "top":
                test_indices = get_train_indices(scores, sub_test_size, test_type)
            elif test_type == "random":
                if sub_train_size >= sub_test_size:
                    ex_indices = train_ind.reshape((TEST_NB_CLASS, sub_train_size))
                    test_indices = ex_indices[:,0:sub_test_size].flatten()
                else:
                    ex_indices = get_test_indices(scores, train_ind, "complet")
                    ex_indices = ex_indices.reshape((TEST_NB_CLASS, -1))
                    #print(ex_indices)
                    for i in range(TEST_NB_CLASS):
                        col_rand_ind = np.arange(len(ex_indices[i]))
                        np.random.shuffle(col_rand_ind)
                        #print("col_rand_ind: ", col_rand_ind)
                        col_rand = ex_indices[i][col_rand_ind[0:sub_test_size-sub_train_size]]
                        test_indices.append(col_rand)
                    test_indices = np.array(test_indices).flatten()
                    test_indices = np.append(test_indices, train_ind)

        train_ind = sorted(train_ind)
        test_indices = sorted(test_indices)


        
        test_data = []
        for t in range(len(TEST_DATA_FOLDER)):
            for i in test_indices:
                _, label, img_path = test_set[t].get_view(i)
                test_data.append([img_path, label])

        test_dataset = IVData_D(
            data=np.array(test_data),
            img_transform=test_img_transform,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        print("Train dataset size: ", len(train_ind))
        print(train_ind)

        test_data_size = len(test_dataset)
        print("Test dataset size: ", test_data_size)
        print(test_indices)


        #load model
        state_dict = model.state_dict()
        for n,p in torch.load(os.path.join(weights_dir, weights_files[n]), map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        #start testing

        model.eval()
        total_test_loss = 0
        total_accuracy = 0.0

        count = 0

        y_gt_labels = np.array([])
        y_predicts = np.array([])
        img_paths = np.array([])

        #compute time

        start_t = time.time()

        with torch.no_grad():
            for data in test_dataloader:
                imgs, labels, paths = data
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE, dtype=torch.int64)
                predicts = model(imgs)
                #loss = loss_fn(predicts, labels)
                #total_test_loss += loss.item()
                accuracy = (predicts.argmax(1) == labels).sum()

                #print("path: ", paths[0])
                #print(len(paths))

                y_gt_labels = np.concatenate((y_gt_labels, labels.cpu().numpy()))
                y_predicts = np.concatenate((y_predicts, predicts.argmax(1).cpu().numpy()))
                img_paths = np.concatenate((img_paths, paths))

                batch_size = imgs.shape[0]
                c = imgs.shape[-3]
                h = imgs.shape[-2]
                w = imgs.shape[-1]
                #if (count+1)%verbose == 0:
                    #print("***---------- batch {} ---------***".format(count+1))
                    #print("gt_label: ", labels.cpu().numpy())
                    #print("predicts: ", predicts.argmax(1).cpu().numpy())
                    #print("accuracy: ", accuracy.cpu().numpy()/batch_size)
                    #if visualized:
                    #    fig, axes = plt.subplots(1, batch_size, figsize=(10*batch_size, 10))
                    #    for i in range(imgs.shape[0]):
                    #        #img = torch.reshape(imgs[i].cpu(), (c, h, w))
                    #        img = imgs[i].cpu().numpy()
                    #        img = np.transpose(img, (1,2,0))
                    #        img = img_normalize(img)
                    #        #print("img shape: ", img.shape)
                    #        axes[i].text(0, 15, 
                    #                    "label: "+ label_names[labels[i].cpu()], 
                    #                    size=30, bbox=dict(boxstyle="square",ec=(1.0, 0.5, 0.5),fc=(1.0, 0.8, 0.8),))
                    #        axes[i].text(0, 40, 
                    #                    "predict: "+ label_names[predicts.argmax(1)[i].cpu()], 
                    #                    size=30, bbox=dict(boxstyle="square",ec=(1.0, 0.5, 0.5),fc=(0.5, 0.8, 0.8),))
                    #        axes[i].imshow(img)
                    #    plt.show()
                
                    
                total_accuracy += accuracy

                count += 1


        end_t = time.time()

        test_time += end_t - start_t

        total_accuracy /= test_data_size
        #print("testset loss: {}".format(total_test_loss))
        print("testset accuracy: {:.6f}".format(total_accuracy))

        y_gt_labels = y_gt_labels.astype(int)
        y_predicts = y_predicts.astype(int)

        #print("img paths: ")
        #print(img_paths)
        #print(img_paths.shape)
        #np.save("res/mlp/img_paths.npy", img_paths)

        #print("grand truth: ")
        #print(y_gt_labels)
        #np.save("res/mlp/y_gt_labels.npy", y_gt_labels)

        #print("predicts: ")
        #print(y_predicts)
        #np.save("res/mlp/y_predicts.npy", y_predicts)

        f1 = compute_f1_global(y_gt_labels, y_predicts)
        f1_c = compute_f1_local(y_gt_labels, y_predicts)
        cm = sklearn.metrics.confusion_matrix(y_gt_labels, y_predicts)

        print("F1 score: {:.6f}".format(f1))
        print("confusion matrix: ")
        print(cm)
        print("F1 score per class: ")
        for l in range(TEST_NB_CLASS):
            print(test_set[0].classes[l], ": ", "{:.6f}".format(f1_c[l]))

        f1_score.append(f1)
        confusion_matrix.append(cm)
        test_img_paths.append(img_paths)
        test_gt_labels.append(y_gt_labels)
        test_pre_labels.append(y_predicts)



    test_time /= n_test
    print("test time cost: {:.6f}".format(test_time))
    print("time cost per img: {:.6f}".format(test_time/test_data_size))

    
    
    f1_mean = np.array(f1_score).mean()
    max_round = np.array(f1_score).argmax()

    print("F1 score: {:.3f}".format(f1_mean))
    print("all test F1-score: ")
    for i in range(len(weights_files)):
        print(weights_files[i],": ", "\t", f1_score[i])


    #path = os.path.join("res/mlp", test_type, TEST_DATA_FOLDER[0].replace('/','_'))
    #if os.path.exists(path):
    #    shutil.rmtree(path)
    #os.makedirs(path)
    #np.save(os.path.join(path, "confusion_matrix.npy"), confusion_matrix[max_round])
    #np.save(os.path.join(path, "test_img_paths.npy"), test_img_paths[max_round])
    #np.save(os.path.join(path, "test_gt_labels.npy"), test_gt_labels[max_round])
    #np.save(os.path.join(path, "test_pre_labels.npy"), test_pre_labels[max_round])
        


if __name__ == '__main__':
    main()
