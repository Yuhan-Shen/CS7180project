# CS7180project

This repository contains the code for the CS7180 course project: Action Segmentation for Instructional Videos with Semi-Weak Supervision.

The data is available on raptor on /mnt/raptor/yuhan/Breakfast_per_task. It can be accessed via Cygnus server.

The environment can be installed via anaconda:
```
conda create --name <env> --file requirements.txt
```
To train the model, please run the following command:
```
python joint_train.py 
```
The default setting is training model on 50% weakly labeled videos with temporal consistency regularization, which does not use unlabeled videos.

To incorporate unlabeled videos for training, please run:
```
python joint_train.py  --dwsa_type mean 
```
or 
```
python joint_train.py  --dwsa_type min
```

For inference, please run:
```
python inference.py
```
or
```
python inference.py --dwsa_type (mean/min)
```
For evaluation, please run:
```
python eval.py --recog_dir results_split1
```
You may change the recog_dir if you save the results in a different folder.
