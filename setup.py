from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='dpsutil',
      version='1.3.1',
      description='This repository contain all util',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/TinDang97/DPS_Util',
      author='TinDang',
      author_email='rainstone1029x@gmail.com',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'redis',
            'confluent_kafka>=1.4.1',
            'lz4',
            'blosc==1.9.1',
            'ffmpeg-python==0.2.0',
            'natsort',
            'attrdict',
            'PyTurboJPEG',
            'opencv-python'
      ],
      python_requires='>=3.6')
