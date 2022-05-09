from setuptools import setup, find_packages

setup(
    name='actuator_dynamics_learn',
    version='0.1',
    packages=find_packages(),
    url='https://bitbucket.org/gsiddhant/actuator_dynamics_learn',
    author='Siddhant Gangapurwala',
    author_email='siddhant@gangapurwala.com',
    python_requires='>=3.6.0',
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'tensorboard',
        'smogn',
        'scikit-learn'
    ]
)
