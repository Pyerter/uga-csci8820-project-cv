# uga-csci8820-project-cv
Group project for computer vision. We are researching NeRF techniques.

## TensoRF

### Running Commands
The key file that will be run in this project for TensoRF is the run_tensorf_synethic.py file. Here's a list of commands that can be run:  

To run the train and test for synthetic (blender) dataset on data folder 0 (the chair folder). "decomp_mode" specifies CP or VM vectorization, with options "cp" or "vm".
```python run_tensorf_synthetic.py --iterations=10000 --data_folder=0 --render_test=True --decomp_mode=cp```
