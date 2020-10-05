from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as rf:
    requirements = rf.read().split()

setup(name='dpsutil',
      version='1.4.1',
      description='This repository contain all util',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/TinDang97/DPS_Util',
      author='TinDang',
      author_email='rainstone1029x@gmail.com',
      packages=find_packages(),
      install_requires=requirements,
      python_requires='>=3.6')
