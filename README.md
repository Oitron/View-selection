# View selection

## Pipeline overview
First execute ***compute_score.py*** file, get different scores with different number of extractions, then execute ***score_analysis.py*** file to save best suitable score. Next use score to run several tests with **MLP** or **KNN**.

---

## Datasets
[Object view dataset](https://drive.google.com/file/d/1CAcdB8Bgq5I2OntWllo1B8hzxGQHCiwt/view?usp=sharing)
**Parameters:**
Classes: 7
Objects: 3
Views: 37

[Component view dataset](https://drive.google.com/file/d/1JROpaByuYgvj6t_U5zKx9oKk4kW_Bi4X/view?usp=sharing)
**Parameters:**
Classes: 7
Objects: 1
Views: 48

---

## Environment
Python: 3.7.3
Pytorch: 1.9.1
CUDA: V11.4.120

Use following command to create environment:
```powershell
conda env create -f env.yaml
```

---

## Configuration
All configurations are in folder ***config***.

***compute_score.json***: Parameters used to calculate scores and analyze them (**compute_score.py** and **score_analysis.py**).  
*"data_set"*: contain dataset path, number of classes, number of objects per class and number of views per object.  
*"nb_tirages"*: maximum number of extraction.  
*"check_points"*: set of check points, when number of extraction hit check point, program will save the score.  
*"nb_sample_set_min & nb_sample_set_max"*: minimum and maximum number of sample classes.  
*"nb_sample_min & nb_sample_max"*: minimum and maximum number of sample views in one class.  
*"score_save_path"*: path to store the scores saved by each check points.  
*"score_analysis_results_save_path"*: path to store the result of analyzing scores (get best round of extraction(optimal global)).  
  
***train_mlp.json***: Parameters used to train model (**train_mlp.py**).  
*"train_set & eval_set"*: contain dataset path, number of classes, number of objects per class and number of views per object.  
*"score_file"*: score record file from the result of analyzing scores.  
*"training_type"*: two options: "top" use optimal views, "random" use random views.  
*"num_execution"*: number of execution, if *"training_type"* set to "top", by default will be 1, if *"training_type"* set to "random", each execution will select different set of random views.  
*"train_size_per_class & eval_size_per_class"*: number of views used for each class in training or evaluation. **Note**: *"eval_size_per_class"* will not be used if *"cross_view"* set to "union", "inter" and "complet".  
*"cross_view"*: Four options: "union": use whole evalset, "inter": use same angles of views in training set, "complet": use views with different angles of views in training set, "other": can manually set the number of *"eval_size_per_class"*.  
*"model_save_dir"*: path to save trained model weights.  
*"hyperparameters"*: **Note**: *"nb_layer_fix"*: set first number of layers fix(frozen parameters). *"cosine_annealing"*: false or true, use cosine annealing algorithm on learning rate or not.  

***test_mlp.json***: Parameters used to test model (**test_mlp.py**).  
*"test_set"*: contain dataset path, number of classes, number of objects per class and number of views per object.  
*"score_file"*: score record file from the result of analyzing scores.  
*"test_size_per_class"*: number of views used for each class in testing. **Note**: will not be used if *"cross_view"* set to "union", "inter" and "complet".  
*"cross_view"*: Four options: "union": use whole testset, "inter": use same angles of views in training set, "complet": use views with different angles of views in training set, "other": can manually set the number of *"test_size_per_class"*.  
*"weights_files_dir"*: folder where store model weights in training process.  
*"nb_layer_fix"*: need to be set to the same value in training process.  
"device": two options: "cuda": use cuda device, "cpu": use cpu device.  

***train&test_knn.json***: Parameters used to train and test on knn (**train&test_knn.py**).  
*"train_set & test_set"*: contain dataset path, number of classes, number of objects per class and number of views per object.  
*"score_file"*: score record file from the result of analyzing scores.  
*"training_type"*: two options: "top" use optimal views, "random" use random views.  
*"num_execution"*: number of execution, if *"training_type"* set to "top", by default will be 1, if *"training_type"* set to "random", each execution will select different set of random views.  
*"train_size_per_class & test_size_per_class"*: number of views used for each class in training or testing. **Note**: *"test_size_per_class"* will not be used if *"cross_view"* set to "union", "inter" and "complet".  
*"cross_view"*: Four options: "union": use whole testset, "inter": use same angles of views in training set, "complet": use views with different angles of views in training set, "other": can manually set the number of *"test_size_per_class"*.  
*"weights_files_dir"*: folder where store model weights in training process or set to "None" if only used as a pre-trained feature extractor.  
*"nb_layer_fix"*: need to be set to the same value in training process, will not be used if *"weights_files_dir"* set to "None".  

---

## Execution
Direct use **python3 + filename**, ex:
```powershell
python3 compute_score.py
```
