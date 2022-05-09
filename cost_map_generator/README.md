# Cost Map Generator

This repository demonstrates the generation of the cost map used in the reward function
for training the RL policies used in the RLOC framework.

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

## Cost Map Generation
You can use the `cost_map.py` script to generate the intermediate images obtained for
generating the cost map.

```
cd $PROJECT_ROOT/scripts
python cost_map.py
```

The generated images will be stored in `$PROJECT_ROOT/results/cost_maps/<current-date-time>/` directory.

## Author(s)
* Siddhant Gangapurwala <siddhant@gangapurwala.com>
