# Height Map Generator

This repository contains the code for generating the height maps used during training.
The code has been adapted for use with: https://www.robots.ox.ac.uk/~mobile/drs/Papers/2021ICRA_gangapurwala.pdf. 

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

## Data Generation
Set the desired parameters for terrain generation and then
run the script
`height_map_generator.py`:
```
cd $PROJECT_ROOT/scripts
python height_map_generator.py
```

## Author(s)
* Siddhant Gangapurwala <siddhant@gangapurwala.com>
