# The main train of fer
# coding: utf-8
import os
import sys
import glob
import json
from base_train import Trainer
#from base_train_double_loss import Trainer
def main(config_path):
	"""
	This is the main function to make the training up
    Parameters:
    -----------
    config_path : path to config file
    """
    # load configs and set random seed
	configs = json.load(open(config_path))
	fer_trainer = Trainer(configs)
	fer_trainer.train()

if __name__ == '__main__':
	config_path = sys.argv[1]
	main(config_path)