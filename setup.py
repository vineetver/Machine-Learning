from setuptools import setup, find_packages

setup(name='Machine Learning',
      version='1.0',
      description='This package contains the code for the Machine Learning repository.',
      author='Vineet Verma',
      author_email='vineetver@hotmail.com',
      packages=find_packages(exclude=['tests']),
      install_requires=[
          'pandas',
          'numpy',
          'matplotlib',
          'sklearn',
          'pandas',
          'scikit-learn',
          'tensorflow',
          'jupyter',
          'seaborn',
          'yfinance',
          'pandas_datareader'
      ]
      )
