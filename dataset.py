import numpy as np
import yaml
import os
import sys

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
config_fp = os.path.join(proj_dir, 'config.yaml')

with open(config_fp, 'r') as f:
    config = yaml.safe_load(f)

class TemporalDataset():
    def __init__(self, data_fp, locations_fp):
        pass

if __name__ == '__main__':
    pass