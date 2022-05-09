# Height Map Encoder

This repository contains the code for training the denoising autoencoder.

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
After generating the heightmaps using the `heightmap_generator` tool, store the dataset
in the `$PROJECT_ROOT/data` directory. To train the network, use the script
`training/denoising_autoencoder.py`:
```
cd $PROJECT_ROOT/scripts
python training/denoising_autoencoder.py
```

## Author(s)
* Siddhant Gangapurwala <siddhant@gangapurwala.com>
