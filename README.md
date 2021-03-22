# STRAug
(Pronounced as "_strog_")

Scene Text Recognition (STR) requires data augmentation functions that are different from object recognition. STRAug offers 36 data augmentation functions that are sorted into 8 groups:

  - Warp - to generate `Curve`, `Distort`, `Stretch` (or Elastic) deformations
  - Geometry - to generate `Perspective`, `Rotation`, `Shrink` deformations
  - Pattern - to create different grids: `Grid`, `VGrid`, `HGrid`, `RectGrid`, `EllipseGrid`
  - Blur - to generate synthetic blur: `GaussianBlur`, `DefocusBlur`, `MotionBlur`, `GlassBlur`, `ZoomBlur`
  - Noise - to add noise: `GaussianNoise`, `ShotNoise`, `ImpulseNoise`, `SpeckleNoise`
  - Weather - to simulate certain weather conditions: `Fog`, `Snow`, `Frost`, `Rain`, `Shadow`
  - Camera - to simulate camera sensor tuning and image compression/resizing: `Contrast`, `Brightness`, `JpegCompression`, `Pixelate`
  - Process - all other image processing issues: `Posterize`, `Solarize`, `Invert`, `Equalize`, `AutoContrast`, `Sharpness`, `Color`

## Run all corruptions on a given image

`python3 test.py --image=<target image>`

For example:

`python3 test.py --image=images/telekom.png `

The corrupted images are in `results` directory.


## References:
  - Image corruptions (eg blur, noise, camera effects, fog, frost, etc) are based on the work of [Hendrycks et al.](https://github.com/hendrycks/robustness)


## Citation
If you find this work useful, please cite:

```
@misc{atienza2021straug,
  title={Data Augmentation for Scene Text Recognition},
  author={Atienza, Rowel},
  year={2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/roatienza/straug}},
}
```

