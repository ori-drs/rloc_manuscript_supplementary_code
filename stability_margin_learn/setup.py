from setuptools import setup, find_packages

setup(
    name='jet_leg_learn',
    version='0.1',
    packages=find_packages(),
    author='Siddhant Gangapurwala',
    author_email='siddhant@gangapurwala.com',
    python_requires='>=2.7.0<3',
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'tensorboard',
        'seaborn',
        'transforms3d',
        'psutil'
    ]
)
