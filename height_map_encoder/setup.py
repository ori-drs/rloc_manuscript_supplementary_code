from setuptools import setup, find_packages

setup(
    name='height_map_encoder',
    version='0.1',
    packages=find_packages(),
    url='https://bitbucket.org/gsiddhant/height_map_encoder',
    author='Siddhant Gangapurwala',
    author_email='siddhant@gangapurwala.com',
    python_requires='>=3.6.0',
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'scikit-image',
        'imageio',
        'pandas',
        'matplotlib',
        'scipy',
        'tensorboard',
    ]
)
