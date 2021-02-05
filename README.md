# robust-str

Many image corruption algorithms are in these repo. The image corruptions (eg blur, noise, camera effects) are base on the work of [Hendrycks](https://github.com/hendrycks/robustness).

## Run all corruption on a given image

`python3 test.py --image=<target image>`

For example:

`python3 test.py --image=images/telekom.png `

The corrupted images are in `results` directory.


