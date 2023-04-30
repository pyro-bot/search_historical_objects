
import torch


DATA_DIR = 'data/hymenoptera_data'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")