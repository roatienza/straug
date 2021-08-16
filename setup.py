from setuptools import setup, find_packages

setup(
  name = 'straug',
  packages = find_packages(),
  include_package_data = True,
  version = '0.1.0',
  license='Apache',
  description='Data Augmentation for STR',
  author = 'Rowel Atienza',
  author_email = 'rowel@eee.upd.edu.ph',
  url = 'https://github.com/roatienza/straug',
  keywords = [
    'computer vision',
    'scene text recognition',
    'ocr',
    'data augmentation'
  ],
  install_requires=[
    'torchvision',
    'magickwand',
    'pillow',
    'opencv-python',
    'opencv-contrib-python',
    'skimage',
    'numpy',
    'Wand'
  ],
  classifiers=[
    'Development Status :: 1 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Computer Vision',
    'License :: OSI Approved :: Apache License',
    'Programming Language :: Python :: 3.9',
  ],
)
