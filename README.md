# Data Augmentation for Scene Text Recognition
(Pronounced as "_strog_")

### Paper


* [ICCV 2021 Workshop](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/papers/Atienza_Data_Augmentation_for_Scene_Text_Recognition_ICCVW_2021_paper.pdf)
* [Arxiv](https://arxiv.org/abs/2108.06949)

## Why it matters?

Scene Text Recognition (STR) requires data augmentation functions that are different from object recognition. STRAug is data augmentation designed for STR. It offers 36 data augmentation functions that are sorted into 8 groups. Each function supports 3 levels or magnitudes of severity or intensity.

Given a source image:

![](/examples/source/delivery.png) 

it can be transformed as follows:

1) `warp.py` - to generate `Curve`, `Distort`, `Stretch` (or Elastic) deformations

`Curve` | `Distort` | `Stretch`
------------ | ------------- | -------------
![](/examples/warp/Curve-2.png) | ![](/examples/warp/Distort-1.png) | ![](/examples/warp/Stretch-1.png)

2) `geometry.py` - to generate `Perspective`, `Rotation`, `Shrink` deformations

`Perspective` | `Rotation` | `Shrink`
------------ | ------------- | -------------
![](/examples/geometry/Perspective-1.png) | ![](/examples/geometry/Rotate-0.png) | ![](/examples/geometry/Shrink-1.png)


3) `pattern.py` - to create different grids: `Grid`, `VGrid`, `HGrid`, `RectGrid`, `EllipseGrid`

`Grid`| `VGrid`| `HGrid` | `RectGrid` | `EllipseGrid`
------------ | ------------- | ------------- | ------------- | -------------
![](/examples/pattern/Grid-0.png) | ![](/examples/pattern/VGrid-0.png) | ![](/examples/pattern/HGrid-0.png) | ![](/examples/pattern/RectGrid-0.png) | ![](/examples/pattern/EllipseGrid-0.png)


4) `blur.py` - to generate synthetic blur: `GaussianBlur`, `DefocusBlur`, `MotionBlur`, `GlassBlur`, `ZoomBlur`

`GaussianBlur` | `DefocusBlur` | `MotionBlur` | `GlassBlur` | `ZoomBlur`
------------ | ------------- | ------------- | ------------- | -------------
![](/examples/blur/GaussianBlur-2.png) | ![](/examples/blur/DefocusBlur-1.png) | ![](/examples/blur/MotionBlur-1.png) | ![](/examples/blur/GlassBlur-1.png) | ![](/examples/blur/ZoomBlur-1.png)


5) `noise.py` - to add noise: `GaussianNoise`, `ShotNoise`, `ImpulseNoise`, `SpeckleNoise`

`GaussianNoise` | `ShotNoise` | `ImpulseNoise` | `SpeckleNoise`
------------ | ------------- | ------------- | ------------- 
![](/examples/noise/GaussianNoise-2.png) | ![](/examples/noise/ShotNoise-2.png) | ![](/examples/noise/ImpulseNoise-2.png) | ![](/examples/noise/SpeckleNoise-2.png) 

6) `weather.py` - to simulate certain weather conditions: `Fog`, `Snow`, `Frost`, `Rain`, `Shadow`

`Fog` | `Snow` | `Frost` | `Rain` | `Shadow`
------------ | ------------- | ------------- | ------------- | -------------
![](/examples/weather/Fog-2.png) | ![](/examples/weather/Snow-1.png) | ![](/examples/weather/Frost-2.png) | ![](/examples/weather/Rain-1.png) | ![](/examples/weather/Shadow-2.png)

7) `camera.py` - to simulate camera sensor tuning and image compression/resizing: `Contrast`, `Brightness`, `JpegCompression`, `Pixelate`

`Contrast` | `Brightness` | `JpegCompression` | `Pixelate`
------------ | ------------- | ------------- | ------------- 
![](/examples/camera/Contrast-2.png) | ![](/examples/camera/Brightness-2.png) | ![](/examples/camera/JpegCompression-2.png) | ![](/examples/camera/Pixelate-2.png) 

8) `process.py` - all other image processing issues: `Posterize`, `Solarize`, `Invert`, `Equalize`, `AutoContrast`, `Sharpness`, `Color`

`Posterize` | `Solarize` | `Invert` | `Equalize`
------------ | ------------- | ------------- | ------------- 
![](/examples/process/Posterize-2.png) | ![](/examples/process/Solarize-2.png) | ![](/examples/process/Invert-2.png) | ![](/examples/process/Equalize-2.png)

`AutoContrast` | `Sharpness` | `Color`
------------ | ------------- | ------------- 
![](/examples/process/AutoContrast-2.png) | ![](/examples/process/Sharpness-2.png) | ![](/examples/process/Color-2.png) 


## Pip install

```
pip3 install straug
```

## How to use

Command line (e.g. input image is `nokia.png`):

```
>>> from straug.warp import Curve
>>> from PIL import Image
>>> img = Image.open("nokia.png")
>>> img = Curve()(img, mag=3)
>>> img.save("curved_nokia.png")
```

Python script (see `test.py`):

`python3 test.py --image=<target image>`

For example:

`python3 test.py --image=images/telekom.png `

The corrupted images are in `results` directory.

*If you want to randomly apply only the desired augmentation types among multiple augmentations, see `test_random_aug.py`*


## Reference
  - Image corruptions (eg blur, noise, camera effects, fog, frost, etc) are based on the work of [Hendrycks et al.](https://github.com/hendrycks/robustness)


## Citation
If you find this work useful, please cite:

```
@inproceedings{atienza2021data,
  title={Data Augmentation for Scene Text Recognition},
  author={Atienza, Rowel},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1561--1570},
  year={2021}
}
```

