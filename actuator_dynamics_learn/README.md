# Actuator Dynamics Learn

This repository contains the code for training the actuator network. Unfortunately,
certain proprietary licenses limit us from sharing the datsets. 

## Setup

This guide assumes that `$PROJECT_ROOT` refers to the absolute path to this repository.

### Installing and Setting Up Virtual Environment
To setup a virtual environment inside the project directory:
```
sudo apt install -y python3-venv
python3 -m venv $PROJECT_ROOT/venv
source $PROJECT_ROOT/venv/bin/activate
```

Now, you are ready to use the virtual environment. To install
the dependencies related to this project, make sure you activate
the environment and then run:

```
cd $PROJECT_ROOT
python install -e .
```

## Training
To resample the dataset and train the network using SMOGN run the script `training/resampled_mlp.py`:
```
cd $PROJECT_ROOT/scripts
python training/resampled_mlp.py
```

## Author(s)
* Siddhant Gangapurwala <siddhant@gangapurwala.com>
