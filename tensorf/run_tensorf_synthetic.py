from src.run_tensorf import train_on_synthetic, test_on_synthetic
from src.utility import fix_base_dir, get_base_dir

import os

if __name__ == "__main__":
    fix_base_dir(os.path.dirname(os.path.realpath(__file__)))
    print(f'PWD: {get_base_dir()}')
    train_on_synthetic()
    #test_on_synthetic()