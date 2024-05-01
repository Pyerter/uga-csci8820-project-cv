# TensoRF

## Environment

The following is alist of the required python pacakges. I used python version 3.11.4 when pip installing each of these.  
- torch
- torchvision
- numpy
- pillow
- imageio
- scipy
- opencv2
- kornia
- lpips (perceptual similarity metric)
- tqdm (progress bar)  

Pip installing [PyTorch](https://pytorch.org/get-started/locally/) will install the first four packages.

## Running Commands
The key file that will be run in this project for TensoRF is the run_tensorf_synethic.py file. Here's a list of commands that can be run:  

To run the train and test for synthetic (blender) dataset on data folder 0 (the chair folder). "decomp_mode" specifies CP or VM vectorization, with options "cp" or "vm". If you wish to check more options, you can check the run_tensorf.py file in the pre_train() function to see all options added to the command.  
```python run_tensorf_synthetic.py --iterations=10000 --data_folder=0 --render_test=True --decomp_mode=cp```

## Contribution

Some code for TensoRF is borrowed from the [original implementation of TensoRF](https://github.com/apchenstu/TensoRF); however, much of it was rewritten in favor of comprehension for myself writing it, structure that I favored, adding comprehensive comments, and/or renaming variables for understanding the material. The files most similar to the original implementation would be the tensoRFVM.py and the util files containing strict mathematical operations. Unlike the paper's code, my code was not as thoroughly tested, so there may be some bugs present.