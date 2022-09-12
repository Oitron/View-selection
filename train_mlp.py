import os
import shutil
import copy
import time

import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from models import DenseNet, InceptionV3, MnasNet, MobileNet, ResNet, SqueezeNet, freeze_features_weights
from add_noise import add_saltPepper_noise, add_gaussian_noise
from load_data import IVData, IVData_F, UnityData, IVData_D

from tools import img_normalize, compute_f1_global, compute_f1_local

from gen_data_indices import get_rand_train_indices, get_train_indices, get_test_indices

from dataset import Dataset

import json






def adjust_learning_rate(lr_int, optimizer, epoch, inter):
    #Set the learning rate to the initial learning rate decayed by 10 every "inter, ex: inter=10" epochs
    lr = lr_int * (0.1 **(epoch//inter))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




MOBILE_NET   = "MOBILENET"
SQUEEZE_NET  = "SQUEEZENET"
MNAS_NET     = "MNASNET"
DENSE_NET    = "DENSENET"
RES_NET      = "RESNET"
INCEPTION_V3 = "INCEPTIONV3"



def main():

    with open("config/train_mlp.json") as json_file:
        variable_dict = json.load(json_file)


    TRAIN_DATA_FOLDER = variable_dict['train_set']['data_path']

    EVAL_DATA_FOLDER = variable_dict['eval_set']['data_path']

    TRAIN_NB_OBJ = variable_dict['train_set']['num_objects']
    TRAIN_NB_VIEW = variable_dict['train_set']['num_views']
    TRAIN_NB_CLASS = variable_dict['train_set']['num_classes']

    EVAL_NB_OBJ = variable_dict['eval_set']['num_objects']
    EVAL_NB_VIEW = variable_dict['eval_set']['num_views']
    EVAL_NB_CLASS = variable_dict['eval_set']['num_classes']

    score_file = variable_dict['score_file']
    num_execution = variable_dict['num_execution']
    cross_view = variable_dict['cross_view']
    sub_train_size = variable_dict['train_size_per_class']
    sub_eval_size = variable_dict['eval_size_per_class']
    train_type = variable_dict['training_type']
    model_save_dir = variable_dict['model_save_dir']

    model_name = variable_dict['hyperparameters']['model'].upper()
    num_epochs = variable_dict['hyperparameters']['num_epochs']
    batch_size = variable_dict['hyperparameters']['batch_size']
    learning_rate = variable_dict['hyperparameters']['learning_rate']
    nb_layer_fix = variable_dict['hyperparameters']['nb_layer_fix']
    cosine_annealing = variable_dict['hyperparameters']['cosine_annealing']


    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device: ", DEVICE)

    if train_type == "top":
        num_execution = 1


    train_img_transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomCrop((700,700)),
        #transforms.ColorJitter(brightness=0.05, hue=0.2),
        transforms.Resize((224, 224)),
        add_saltPepper_noise(0.05, 0.2),
        add_gaussian_noise(0.0, 1.0, 10, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    eval_img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        #add_saltPepper_noise(0.05, 0.3),
        #add_gaussian_noise(0.0, 1.0, 10, 0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    if model_name == INCEPTION_V3:
        train_img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            #add_saltPepper_noise(0.05, 0.3),
            #add_gaussian_noise(0.0, 1.0, 10, 0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    #default
    if model_name == MOBILE_NET:
        model = MobileNet(num_classes=TRAIN_NB_CLASS, nb_layer_fix=nb_layer_fix).to(DEVICE)
        print("Using model: ", MOBILE_NET)
    elif model_name == SQUEEZE_NET:
        model = SqueezeNet(num_classes=TRAIN_NB_CLASS).to(DEVICE)
        print("Using model: ", SQUEEZE_NET)
    elif model_name == MNAS_NET:
        model = MnasNet(num_classes=TRAIN_NB_CLASS).to(DEVICE)
        print("Using model: ", MNAS_NET)
    elif model_name == DENSE_NET:
        model = DenseNet(num_classes=TRAIN_NB_CLASS).to(DEVICE)
        print("Using model: ", DENSE_NET)
    elif model_name == RES_NET:
        model = ResNet(num_classes=TRAIN_NB_CLASS).to(DEVICE)
        print("Using model: ", RES_NET)
    elif model_name == INCEPTION_V3:
        model = InceptionV3(num_classes=TRAIN_NB_CLASS).to(DEVICE)
        print("Using model: ", INCEPTION_V3)

    #freeze features_weights
    freeze_features_weights(model, ('features'))
    #loss function
    loss_fn = CrossEntropyLoss().to(DEVICE)
    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    

    #model save
    root = os.path.join(model_save_dir, "trainSize_"+str(sub_train_size), train_type)
    #if os.path.exists(root):
    #    shutil.rmtree(root)
    models_save_dir = os.path.join(root, "models")
    indices_save_dir = os.path.join(root, "train_indices")
    if not os.path.exists(root):
        os.makedirs(models_save_dir)
        os.makedirs(indices_save_dir)




    mean_f1_score = 0
    for round in range(num_execution):
        # get indices
        train_set = []
        for l in range(len(TRAIN_DATA_FOLDER)): 
            train_set.append(
                Dataset(
                    root_folder=TRAIN_DATA_FOLDER[l],
                    nb_class=TRAIN_NB_CLASS,
                    nb_obj=TRAIN_NB_OBJ,
                    nb_view=TRAIN_NB_VIEW,
                )
            )

        eval_set = []
        for l in range(len(EVAL_DATA_FOLDER)): 
            eval_set.append(
                Dataset(
                    root_folder=EVAL_DATA_FOLDER[l],
                    nb_class=EVAL_NB_CLASS,
                    nb_obj=EVAL_NB_OBJ,
                    nb_view=EVAL_NB_VIEW,
                )
            )

        scores = np.load(score_file)
        scores = scores.reshape((TRAIN_NB_CLASS, TRAIN_NB_OBJ, TRAIN_NB_VIEW))
        
        train_indices = []
        eval_indices = []
        if cross_view != "other":
            train_indices = get_train_indices(scores, sub_train_size, type=train_type)
            eval_indices = get_test_indices(scores, train_indices, cross_view)
        else:
            if train_type == "top":
                train_indices = get_train_indices(scores, sub_train_size, type=train_type)
                ex_indices = get_train_indices(scores, sub_eval_size, type=train_type)
                eval_indices = get_test_indices(scores, ex_indices, "inter")
            elif train_type == "random":
                if sub_train_size >= sub_eval_size:
                    train_indices = get_train_indices(scores, sub_train_size, type=train_type)
                    ex_indices = train_indices.reshape((TRAIN_NB_CLASS, sub_train_size))
                    eval_indices = ex_indices[:,0:sub_eval_size].flatten()
                else:
                    eval_indices = get_train_indices(scores, sub_eval_size, type=train_type)
                    ex_indices = eval_indices.reshape((TRAIN_NB_CLASS, sub_eval_size))
                    train_indices = ex_indices[:,0:sub_train_size].flatten()

        train_indices = sorted(train_indices)
        eval_indices = sorted(eval_indices)



        train_data = []
        eval_data = []
        for t in range(len(TRAIN_DATA_FOLDER)):
            for i in train_indices:
                _, label, img_path = train_set[t].get_view(i)
                train_data.append([img_path, label])
        for t in range(len(EVAL_DATA_FOLDER)):
            for i in eval_indices:
                _, label, img_path = eval_set[t].get_view(i)
                eval_data.append([img_path, label])

        print("train data shape: ", np.array(train_data).shape)


        #load data
        train_dataset = IVData_D(
            data=np.array(train_data),
            img_transform=train_img_transform,
        )
        eval_dataset = IVData_D(
            data=np.array(eval_data),
            img_transform=eval_img_transform,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        #data size
        train_data_size = len(train_dataset)
        eval_data_size = len(eval_dataset)

        print("Train dataset size: ", train_data_size)
        print(train_indices)
        print("Eval dataset size: ", eval_data_size)
        print(eval_indices)

        #start training
        #from torch.utils.tensorboard import SummaryWriter

        #set useful values
        total_train_step = 0
        total_val_step = 0
        best_accuracy = 0.0
        best_epoch = 0

        best_weights = copy.deepcopy(model.state_dict())

        #visualize
        #writer = SummaryWriter("learning_log")

        best_f1_score = 0

        #compute time
        start_t = time.time()

        for i in range(num_epochs):
            if cosine_annealing:
                scheduler.step(i)
            print("************----------------{}/{} execution,  {}/{} epoch ----------------************".format(round+1, num_execution, i+1, num_epochs))
            #print current learning rate
            print("learning rate: {:.7f}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            #training
            model.train()
            for data in train_dataloader:
                imgs, labels, _ = data
                #print("img shape:", imgs.shape)
                #print("label shape:", len(labels))
                #print("label: ", labels)
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE, dtype=torch.int64)
                predicts = model(imgs)
                optimizer.zero_grad()
                #forward + backward
                loss = loss_fn(predicts, labels)
                loss.backward()
                optimizer.step()

                total_train_step += 1
                if total_train_step%100 == 0:
                    print("train step: {}, Loss: {}".format(total_train_step, loss.item()))
                    #writer.add_scalar("train loss", loss.item(), total_train_step)

            #validing
            model.eval()
            total_eval_loss = 0.0
            total_accuracy = 0.0

            y_gt_labels = np.array([])
            y_predicts = np.array([])

            with torch.no_grad():
                for data in eval_dataloader:
                    imgs, labels, _ = data
                    imgs = imgs.to(DEVICE)
                    labels = labels.to(DEVICE, dtype=torch.int64)
                    predicts = model(imgs)
                    loss = loss_fn(predicts, labels)
                    total_eval_loss += loss.item()
                    accuracy = (predicts.argmax(1) == labels).sum()
                    #print("predicts: ", predicts.argmax(1))
                    #print("gt_label: ", labels)
                    #print("accuracy: ", accuracy)
                    total_accuracy += accuracy

                    y_gt_labels = np.concatenate((y_gt_labels, labels.cpu().numpy()))
                    y_predicts = np.concatenate((y_predicts, predicts.argmax(1).cpu().numpy()))

            total_accuracy /= eval_data_size
            print("evalset loss: {}".format(total_eval_loss))
            print("evalset accuracy: {}".format(total_accuracy))
            #writer.add_scalar("eval loss", total_eval_loss, total_val_step)
            #writer.add_scalar("eval accuracy", total_accuracy, total_val_step)

            y_gt_labels = y_gt_labels.astype(int)
            y_predicts = y_predicts.astype(int)

            f1 = compute_f1_global(y_gt_labels, y_predicts)
            f1_c = compute_f1_local(y_gt_labels, y_predicts)

            print("F1 score: {:.6f}".format(f1))
            print("F1 score per class: ")
            for l in range(TRAIN_NB_CLASS):
                print(train_set[0].classes[l], ": ", "{:.6f}".format(f1_c[l]))

            total_val_step += 1

            #save best model weights
            if best_accuracy < total_accuracy:
                best_accuracy = total_accuracy
                best_epoch = i+1
                best_weights = copy.deepcopy(model.state_dict())

            if best_f1_score < f1:
                best_f1_score = f1

        end_t = time.time()
        time_cost = end_t - start_t

        print("time cost: ", time_cost)
        #close visualize
        #writer.close()

        mean_f1_score += best_f1_score

        #save model
        print("best epoch: {}, accuracy: {:.3f}".format(best_epoch, best_accuracy))

        
        torch.save(best_weights, os.path.join(models_save_dir, "model_{}_layer_{}_trainSize_{}_numExe_{}_best_{}.pth".format(train_type, nb_layer_fix, sub_train_size, round, best_epoch)))
        np.save(os.path.join(indices_save_dir, "train_indices_{}_layer_{}_trainSize_{}_numExe_{}".format(train_type, nb_layer_fix, sub_train_size, round)), train_indices)
        print("best model saved")

    mean_f1_score /= num_execution
    print("F1_score: {:.3f}".format(mean_f1_score))







        





if __name__ == '__main__':
    main()

    
    
