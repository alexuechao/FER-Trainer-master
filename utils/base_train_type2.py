# Trainer of Facial Expression Recongition for Pytorch
# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import argparse
import logging
import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils import *
from utils.utils import *
from utils.utils import get_model
from utils.mixup import *
from dataloader import Dataloader
from models import *
from models.peleenet import *
from models.resnet_cut import *

class Trainer(object):
	def __init__(self, config):
		self.config = config

	def get_logger(self, model, task_name, path):
		log_format = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
		log_file = '{}_{}.log'.format(model, task_name)
		log_path = os.path.join(path, log_file)
		# make logger
		if log_path is not None:
			if os.path.exists(log_path):
				os.remove(log_path)
			log_dir = os.path.dirname(log_path)
			if not os.path.exists(log_dir):
				os.makedirs(log_dir)
			# get logger
			logger = logging.getLogger()
			logger.handlers = []
			formatter = logging.Formatter(log_format)
			# file handler
			handler = logging.FileHandler(log_path)
			handler.setFormatter(formatter)
			logger.addHandler(handler)
			# stream handler
			handler = logging.StreamHandler()
			handler.setFormatter(formatter)
			logger.addHandler(handler)
			logger.setLevel(logging.INFO)
			logging.basicConfig(level=logging.INFO, format=log_format)
		else:
			logger = logging.getLogger()
			logger.setLevel(logging.INFO)
			logging.basicConfig(level=logging.INFO, format=log_format)
		return logger

	def save_mix_images(self, x_batch, end_epoch = 0, end_batch_idx = 5):
		if epoch == end_epoch and batch_idx <= end_batch_idx:
			for i in range(len(x_batch)):
				unloader = transforms.ToPILImage()
				image = x_batch[i].cpu().clone()
				image = image.squeeze(0)
				image = unloader(image)
				out_dir = 'mix_results'
				if not os.path.exists(out_dir):
					os.makedirs(out_dir)
				image.save('mix_results/results_{}_{}.jpg'.format(batch_idx, i))

	def eval_model(self, Valloader, criterion, net, epoch, logger, writer):
		global Val_acc
		global best_Val_acc
		global best_Val_acc_epoch
		use_cuda = torch.cuda.is_available()
		net.eval()
		Val_loss = 0
		correct = 0
		total = 0
		for batch_idx, (inputs, targets) in enmuerate(Valloader):
			if use_cuda:
				inputs, targets = inputs.cuda(device = self.config.device_ids[0]),\
				                  targets.cuda(device = self.config.device_ids[0])
			inputs, targets = Variable(inputs), Variable(targets)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			Val_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
		Val_acc = 100. * float(correct)/total
		Val_epoch_loss = Val_loss / (batch_idx + 1)
		cur_step = epoch + 1
		save_model_path = os.path.join(self.config.path,'fer_model_epoch_{}.t7'.format(cur_step))
		cur_state = {
		    'net': net.state_dict() if use_cuda else net,
		    'acc': Val_acc,
		    'epoch': cur_step,
		}
		torch.save(cur_state, save_model_path)
		epoch_log = ('Epoch[%d] Val-Accuracy: %.4f, Val-loss: %.4f' %
			        (cur_step, Val_acc, Val_epoch_loss))
		logger.info(epoch_log)
		writer.add_scalar('Accuracy/Val', Val_acc, cur_step)
		writer.add_scalar('Loss/Val', Val_epoch_loss, cur_step)
		if Val_acc > best_Val_acc:
			print("Saving..")
			print("best_Val_acc: %0.3f" % Val_acc)
			best_val_acc = ("Best_Val_acc: %0.3f" % Val_acc)
			logger.info(best_val_acc)
			state = {
			    'net' : net.state_dict() if use_cuda else net,
			    'best_val_acc' : Val_acc,
			    'best_val_epoch' : epoch,
			}
			torch.save(state, os.path.join(self.config.path, 'Val_model.t7'))
			best_Val_acc = Val_acc
			best_Val_acc_epoch = epoch

	def test_model(self, Testloader, criterion, net, epoch, logger, writer):
		global Test_acc
		global best_Test_acc
		global best_Test_acc_epoch
		use_cuda = torch.cuda.is_available()
		net.eval()
		Test_loss = 0
		correct = 0
		total = 0
		for batch_idx, (inputs, targets) in enmuerate(Testloader):
			if use_cuda:
				inputs, targets = inputs.cuda(device = self.config.device_ids[0]),\
				                  targets.cuda(device = self.config.device_ids[0])
			inputs, targets = Variable(inputs), Variable(targets)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			Test_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
		## logging.
		Test_acc = 100.*float(correct)/total
		Test_epoch_loss = Test_loss / (batch_idx + 1)
		epoch_log = ('Epoch[%d] Test-Accuracy: %.4f, Test-Loss: %.4f'%
			        (epoch + 1, Test_acc, Test_epoch_loss))
		logger.info(epoch_log)
		writer.add_scalar('Accuracy/test', Test_acc, epoch + 1)
		writer.add_scalar('Loss/test', Test_epoch_loss, epoch + 1)
		## save checkpoint
		if Test_acc > best_Test_acc:
			print('Saving..')
			print('best_Test_acc: %0.3f' % Test_acc)
			best_test_acc = ('Best_Test_Accuracy: %0.3f' Test_acc)
			logger.info(best_test_acc)
			state = {
			    'net': net.state_dict() if use_cuda else net,
			    'best_Test_acc': best_Test_acc,
			    'best_Test_acc_epoch': epoch,
			}
			if not os.path.isdir(self.config.path):
				os.mkdir(self.config.path)
			torch.save(state, os.path.join(self.config.path, 'Test_model.t7'))
			best_Test_acc = Test_acc
			best_Test_acc_epoch = epoch

	def train_model(self, net, criterion, optimizer, scheduler, trainloader, valloader, testloader, logger, writer):
		use_cuda = torch.cuda.is_available()
		## if or not pre
		if self.config.pretrained:
			checkpoint = torch.load(os.path.join(self.config.pretrained))
			net.load_state_dict(checkpoint['net'])
			print("==> Loaded checkpoint from pretrained model-'{}'".format(pretrained))
		## resume_from
		elif self.config.resume_from:
			print('Loading weight...')
			checkpoint = torch.load(os.path.join(resume_from))
			net.load_state_dict(checkpoint['net'])
			acc = checkpoint['acc']
			cur_epoch = checkpoint['epoch']
			current_epoch = cur_epoch
			print("=> Loaded checkpoint='{}' (epoch={})".format(resume_from, current_epoch))
		else:
			resume_from = 0
			current_epoch = 0
			print('==> Building model..')

		idx_ter = 0
		for epoch in range(current_epoch, self.config.num_epochs):
			print('Epoch {}/{}'.format(epoch, self.config.num_epochs))
			global Train_acc
			net.train()
			train_loss = 0
			correct = 0
			total = 0
			current_lr = optimizer.param_groups[0]['lr']
			#warmup_steps = 500
			for batch_idx, (inputs, targets) in enmuerate(trainloader):
				## warmup
				if batch_idx <= self.config.warmup_steps and epoch == 0:
					warmup_percent_done = (batch_idx + 1) / (warmup_steps + 1)
					warmup_lr = float(self.config.init_lr * warmup_percent_done)
					current_lr = warmup_lr
					set_lr(optimizer, current_lr)
				else:
					current_lr = current_lr
				if use_cuda:
					inputs, targets = inputs.cuda(device = self.config.device_ids[0]),\
					                  targets.cuda(device = self.config.device_ids[0])
				inputs, targets = Variable(inputs), Variable(targets)
				## mixup or cutmix
				if self.config.mixup or self.config.cutmix:
					if epoch <=self.config.num_epochs - 20:
						if self.config.mixup and (not self.config.cutmix):
							x_batch, y_batch_a, y_batch_b, lam = mixup_data_radio(inputs, targets, alpha = 0.25, mix_radio = 0.5)
						elif self.config.cutmix and (not self.config.mixup):
							x_batch, y_batch_a, y_batch_b, lam = cutmix_data_radio(inputs, targets, alpha = 0.25, mix_radio = 0.5)
						elif self.config.mixup and self.config.cutmix:
							x_batch, y_batch_a, y_batch_b, lam = cutmix_data_radio(inputs, targets, alpha=0.15, mix_radio=0.5)\
							if np.random.rand() > 0.5 else mixup_data_radio(inputs, targets, alpha=0.15, mix_radio=0.5)
						## save mix_img
						if self.config.save_mix_results:
							self.save_mix_images(x_batch = x_batch, end_epoch = 0, end_batch_idx = 5)
						outputs = net(x_batch.cuda())
						loss = mixup_criterion(criterion, outputs, y_batch_a.cuda(device=device_ids[0]), y_batch_b.cuda(device=device_ids[0]), lam)
						optimizer.zero_grad()
						loss.backward()
						utils.clip_gradient(optimizer, 0.1)
						optimizer.step()
						train_loss += loss.item()
						_,predicted = torch.max(outputs.data, 1)
						total += targets.size(0)
						lam = torch.tensor(lam, dtype=torch.float32)
						correct += lam * predicted.eq(y_batch_a.data).cpu().sum() + (1 - lam) * predicted.eq(y_batch_b.data).cpu().sum()
					else:
						optimizer.zero_grad()
						outputs = net(inputs)
						loss = criterion(outputs, targets)
						loss.backward()
						utils.clip_gradient(optimizer, 0.1)
						optimizer.step()
						train_loss += loss.item()
						_, predicted = torch.max(outputs.data, 1)
						total += targets.size(0)
						correct += predicted.eq(targets.data).cpu().sum()
				else:
	                optimizer.zero_grad()
	                outputs = net(inputs)
	                loss = criterion(outputs, targets)
	                loss.backward()
	                utils.clip_gradient(optimizer, 0.1)
	                optimizer.step()
	                train_loss += loss.item()
	                _, predicted = torch.max(outputs.data, 1)
	                total += targets.size(0)
	                correct += predicted.eq(targets.data).cpu().sum()

	            accuracy = 100. * float(correct)/total
	            ##logging
	            if batch_idx % 20 == 0:
	            	train_log = ('Epoch[%d/%d] Batch[%d] lr: %.6f, Training_Loss=%.4f, Train_Accuracy=%.4f' %
	            		(epoch + 1, num_epochs, batch_idx, current_lr, train_loss / (batch_idx + 1), accuracy))
	            	logger.info(train_log)
	            	idx_ter += 1
	            	writer.add_scalar('Accuracy/iter_train', accuracy, idx_ter)
	            	writer.add_scalar('Loss/iter_train', train_loss / (batch_idx + 1), idx_ter)
	        scheduler.step()
	        train_acc = 100.*correct/total
	        train_epoch_loss = train_loss / (batch_idx+1)
	        epoch_log = ('Epoch[%d] Training-Accuracy=%.4f, Train-Loss=%.4f'%(epoch + 1, train_acc, train_epoch_loss))
	        logger.info(epoch_log)
	        writer.add_scalar('Accuracy/epoch_train', accuracy, epoch+1)
	        writer.add_scalar('Loss/epoch_train', train_epoch_loss, epoch+1)
	        ## val or test
	        if not self.self.config.evel_only:
				print('Starting valing..')
				self.eval_model(valloader, criterion, net, epoch, logger, writer)
				print('Starting testing...')
				self.test_model(testloader, criterion, net, epoch, logger, writer)
			else:
			    print('Only valing..')
			    self.eval_model(valloader, criterion, net, epoch, logger, writer)

	def trainer(self):
		best_Val_acc = 0
		best_Val_acc_epoch = 0
		best_Test_acc = 0
		best_Test_acc_epoch = 0
		use_cuda = torch.cuda.is_available()
		## get logger
		logger = getLogger(self.config.model_name, self.config.task_name, self.config.path)
		logger.info("Task Name : {}".format(self.config.task_name))
	    logger.info("Backbone_name : {}".format(self.config.model_name))
	    logger.info("input_shape : (3,{}.{})".format(self.config.input_shape,self.config.input_shape))
	    logger.info("num_epochs : {}".format(self.config.num_epochs))
	    logger.info("resume_from : {}".format(self.config.resume_from))
	    logger.info("pretrained : {}".format(self.config.pretrained))
	    ## tensorboard writer
	    log_dir = os.path.join(self.config.path,"{}".format("tensorboard_log"))
	    if not os.path.isdir(log_dir):
	    	os.mkdir(log_dir)
	    writer = SummaryWriter(log_dir)
	    ## get model of train
	    net = get_model(self.config.model_name)
	    net = torch.nn.DataParallel(net, device_ids = self.config.device_ids)
	    net = net.cuda(device = device_ids[0])
	    ## loss
	    criterion = nn.CrossEntropyLoss()
	    ## optimizer
	    if self.config.optimizers == 'SGD':
	    	optimizer = optim.SGD(net.parameters(), lr=self.config.init_lr, momentum=0.9, weight_decay=self.config.weight_decay)
	    elif self.config.optimizers == 'Adam':
	    	optimizer = optim.Adam(net.parameters(), lr=self.config.init_lr, weight_decay=self.config.weight_decay)
	    milestones = [80,150,200,300]
	    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
	    logger.info(("============opti==========="))
	    logger.info("Optimizer:{}".format(self.config.optimizers))
	    logger.info("lr:{}".format(self.config.init_lr))
	    logger.info("weight_decay:{}".format(self.config.weight_decay))
	    logger.info("lr_scheduler: MultiStepLR")
	    logger.info("milestones:{}".format(milestones))
	    ## augumation
	    normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
		transform_train = transforms.Compose([
		    transforms.RandomCrop(self.config.input_shape),
		    transforms.RandomHorizontalFlip(),
		    transforms.RandomBrightness(brightness = self.config.brightness, brightness_ratio=self.config.brightness_ratio),
		    transforms.RandomBlur(blur_ratio = self.config.blur_ratio),
		    transforms.RandomRotation(degrees = self.config.degrees, rotation_ratio = 0.1),
		    transforms.ColorJitter(brightness = self.config.color_brightnesss, contrast = self.config.color_contrast,\
		                           saturation = self.config.color_saturation, hue=0),
		    transforms.ToTensor(),
		    #normalize,
		])
		## test aug
		transform_test = transforms.Compose([
		    transforms.CenterCrop(input_shape),
		    transforms.ToTensor(),
		    #normalize,
		])
        logger.info(("============opti==========="))
		logger.info("brightness:{}".format(self.config.brightness))
		logger.info("brightness_ratio:{}".format(self.config.brightness_ratio))
		logger.info("blur_ratio:{}".format(self.config.blur_ratio))
		logger.info("degrees:{}".format(self.config.degrees))
		logger.info("color_brightnesss:{}".format(self.config.color_brightnesss))
		logger.info("color_contrast:{}".format(self.config.color_contrast))
		logger.info("color_saturation:{}".format(self.config.color_saturation))
        ## prepara data
        print('==> Preparing data..')
		trainset = DataLoader(split = 'Training', transform=transform_train)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.config.batch_size * len(self.config.device_ids), shuffle=True)
		Valset = DataLoader(split = 'Valing', transform=transform_test)
		Valloader = torch.utils.data.DataLoader(Valset, batch_size=64 * len(self.config.device_ids), shuffle=False)
		Testset = DataLoader(split = 'Testing', transform=transform_test)
		Testloader = torch.utils.data.DataLoader(Testset, batch_size=64 * len(self.config.device_ids), shuffle=False)
        ## train
		logger.info(("====== Training !!!======"))
		train_model(net, criterion, optimizer, scheduler, trainloader, valloader, testloader, logger, writer)
		logger.info(("======Finsh Training !!!======"))    
		logger.info(("best_val_acc_epoch: %d, best_val_acc: %0.3f" % (best_Val_acc_epoch, best_Val_acc)))
		logger.info(("best_test_acc_epoch: %d, best_test_acc: %0.3f" % (best_Test_acc_epoch, best_Test_acc)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trainer of Facial Expression Recongition for Pytorch')
    parser.add_argument('--model_name', type=str, default='Resnet18', help='CNN architecture')
    parser.add_argument('--task_name', type=str, default='FER_cnn', help='CNN architecture')
    parser.add_argument('--batch_size', default=128, type=int, help='learning rate')
    parser.add_argument('--init_lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--input_shape', default=64, type=int, help='RandomCrop size')
    parser.add_argument('--num_epochs', default=230, type=float, help='Number epochs')
    parser.add_argument('--resume_from', default=None, type=str, help='path to latest checkpoint')
    parser.add_argument('--pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('--eval_only', type=bool, default=False, help='eval_only or not')
    parser.add_argument('--optimizers', type=str, default='SGD', help='Train Optimizer')
    parser.add_argument('--mixup', type=int, default=1)
    parser.add_argument('--cutmix', type=int, default=1)
    parser.add_argument('--save_mix_results', type=int, default=1)
    config = parser.parse_args()

    fer_trainer = Trainer(config)
    fer_trainer.trainer()
