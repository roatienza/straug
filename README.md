# robust-str

Many image corruption algorithms are in this repo. The image corruptions (eg blur, noise, camera effects) are based on the work of [Hendrycks et al.](https://github.com/hendrycks/robustness) on ImageNet-C. Image distortion and compression using anchor points are based on the work of [Luo et al.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_Learn_to_Augment_Joint_Data_Augmentation_and_Network_Optimization_for_CVPR_2020_paper.pdf)

## Run all corruptions on a given image

`python3 test.py --image=<target image>`

For example:

`python3 test.py --image=images/telekom.png `

The corrupted images are in `results` directory.


