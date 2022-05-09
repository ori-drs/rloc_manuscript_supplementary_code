from setuptools import setup, find_packages

setup(
    name='cost_map_generator',
    version='0.1',
    packages=find_packages(),
    url='https://bitbucket.org/gsiddhant/cost_map_generator',
    author='Siddhant Gangapurwala',
    author_email='siddhant@gangapurwala.com',
    python_requires='>=3.6.0',
    install_requires=[
        'numpy',
        'scikit-image',
        'imageio',
        'matplotlib',
        'pypng',
        'pandas',
        'seaborn'
    ]
)
