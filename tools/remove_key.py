import os
import sys
import argparse
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def module_key_remove(model_path, out_path):
	model_name = model_path.split('/')[-1]
	#import pdb
	#pdb.set_trace()
	model_name = model_name[:-3]
	checkpoint = torch.load(model_path, map_location = 'cpu')
	out_dic = {k.replace('module.',''):v for k,v in checkpoint['net'].items()}
	save_model_path = os.path.join(out_path,'fer_{}.t7'.format(model_name))
	torch.save(out_dic, save_model_path)

if __name__ == '__main__':
	model_path = sys.argv[1]
	out_path = sys.argv[2]
	module_key_remove(model_path, out_path)