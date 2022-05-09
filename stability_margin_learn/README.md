# Jet Leg Learn

This repository contains the dataset generation and network
training code for the stability margin network detailed in the
work of [Orsolino. et al. 2021](https://www.robots.ox.ac.uk/~mobile/drs/Papers/2021IROS_orsolino.pdf).


## Setup

This guide assumes that `$PROJECT_ROOT` refers to the absolute path to this repository.

### Installing and Setting Up Virtual Environment
To setup a virtual environment inside the project directory:
```
sudo apt install -y python-venv
python -m venv $PROJECT_ROOT/venv
source $PROJECT_ROOT/venv/bin/activate
```

Now, you are ready to use the virtual environment. To install
the dependencies related to this project, make sure you activate
the environment and then run:

```
cd $PROJECT_ROOT/jet_leg_common
python install -e .
```

And then
```
cd $PROJECT_ROOT
pip install -e .
```

## Data Generation
If everything went well, you should be able to run the data generation script
`data_generation/stability_margin.py`, assuming you have activated the virtual environment,
like so:
```
cd $PROJECT_ROOT/scripts
python data_generation/stability_margin.py
```
The generated dataset will be stored in the `$PROJECT_ROOT/data` directory.

## Training
After generating the training dataset, you can start the training using the script
`training/stability_margin.py`:
```
cd $PROJECT_ROOT/scripts
python training/stability_margin.py
```

## Deployment
Use the script 
`utility/model_to_txt_converter.py` to save the trained model parameters to a txt file for
use with the custom C++ MLP implementation:
```
cd $PROJECT_ROOT/scripts
python utility/model_to_txt_converter.py
```


## Author(s)
* Siddhant Gangapurwala <siddhant@gangapurwala.com>
