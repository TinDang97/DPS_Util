from setuptools import setup, find_packages

setup(name='dpsutil',
      version='0.1',
      description='This repository contain all util like compression, redis, kafka (with custom config),'
                  ' logging_tool',
      url='https://github.com/TinDang97/DPS_Util',
      author='TinDang',
      author_email='rainstone1029x@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'scikit-build',
            'numpy',
            'redis',
            'kafka-python',
            'lz4',
            'blosc',
            'ffmpeg-python', 'natsort'
      ],
      zip_safe=False)
