from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='dpsutil',
      version='1.0.0',
      description='This repository contain all util',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/TinDang97/DPS_Util',
      author='TinDang',
      author_email='rainstone1029x@gmail.com',
      packages=find_packages(),
      install_requires=[
            'scikit-build',
            'cmake>3.12',
            'numpy',
            'redis',
            'kafka-python',
            'lz4',
            'blosc',
            'ffmpeg-python',
            'natsort'
      ],
      python_requires='>=3.6',
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ])
