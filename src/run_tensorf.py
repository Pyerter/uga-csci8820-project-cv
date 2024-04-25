from .data_loading.data_synthetic import SyntheticSet, DATA_FOLDERS
from .tensorf.tensoRFCP import TensoRFCP

def train_on_synthetic():
    dataset = SyntheticSet(DATA_FOLDERS[0])
    model = TensoRFCP()

if __name__ == "__main__":
    train_on_synthetic()