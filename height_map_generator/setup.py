from setuptools import setup, find_packages

setup(name='height_map_generator',
      version='0.0.1',
      packages=find_packages(),
      url='https://bitbucket.org/gsiddhant/height_map_generator',
      author='Siddhant Gangapurwala',
      author_email='siddhant@gangapurwala.com',
      python_requires='>=3.6.0,<3.7.0',
      install_requires=[
          'numpy',
          'matplotlib',
          'imageio',
          'scipy',
          'pypng',
          'tqdm'
      ]
      )
