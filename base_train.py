# Trainer of Facial Expression Recongition for Pytorch
# coding: utf-8
import os
import sys
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
#from utils.utils import get_model, set_lr, clip_gradient
from utils.mixup import *
from dataloader import DataLoader
import transforms as transforms

class Trainer(object):
	def __init__(self, configs):
		self.configs = configs
		self.train_datasets = self.configs['train_datasets']
		self.val_datasets = self.configs['val_datasets']
		self.test_datasets = self.configs['test_datasets']
		self.device_ids = self.configs['gpu_device_ids']
		self.task_name = self.configs['task_name']
		self.model_name = self.configs['model_name']
		self.job_id = self.configs['job_id']
		self.input_shape = self.configs['input_shape']
		self.batch_size = self.configs['batch_size']
		self.resume_from = self.configs['resume_from']
		self.pretrained = self.configs['pretrained']
		self.num_epochs = self.configs['num_epochs']
		self.warmup_steps = self.configs['warmup_steps']
		self.init_lr = self.configs['init_lr']
		self.mixup = self.configs['mixup']
		self.cutmix = self.configs['cutmix']
		self.optimizers = self.configs['optimizer']
		self.weight_decay = self.configs['weight_decay']
		self.brightness = self.configs['brightness']
		self.brightness_ratio = self.configs['brightness_ratio']
		self.blur_ratio = self.configs['blur_ratio']
		self.degrees = self.configs['degrees']
		self.color_brightnesss = self.configs['color_brightnesss']
		self.color_contrast = self.configs['color_contrast']
		self.color_saturation = self.configs['color_saturation']
		self.save_mix_results = self.configs['save_mix_results']
		self.evel_only = self.configs['evel_only']
        
		self.Val_acc = 0
		self.best_Val_acc = 0
		self.best_Val_acc_epoch = 0
		self.Test_acc = 0
		self.best_Test_acc = 0
		self.best_Test_acc_epoch = 0

	def get_logger(self, model_name, task_name, job_id, path):
		log_format = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
		log_file = '{}_{}_{}.log'.format(model_name, task_name, job_id)
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

	def save_mix_images(self, x_batch, epoch, batch_idx, end_epoch = 0, end_batch_idx = 5):
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

	def eval_model(self, Valloader, criterion, net, epoch, logger, writer, path):
		#global Val_acc
		#global best_Val_acc
		#global best_Val_acc_epoch
		use_cuda = torch.cuda.is_available()
		net.eval()
		Val_loss = 0
		correct = 0
		total = 0
		for batch_idx, (inputs, targets) in enumerate(Valloader):
			if use_cuda:
				inputs, targets = inputs.cuda(device = self.device_ids[0]),\
				                  targets.cuda(device = self.device_ids[0])
			inputs, targets = Variable(inputs), Variable(targets)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			Val_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
		self.Val_acc = 100. * float(correct)/total
		Val_epoch_loss = Val_loss / (batch_idx + 1)
		cur_step = epoch + 1
		save_model_path = os.path.join(path,'fer_model_epoch_{}.t7'.format(cur_step))
		cur_state = {
		    'net': net.state_dict() if use_cuda else net,
		    'acc': self.Val_acc,
		    'epoch': cur_step,
		}
		torch.save(cur_state, save_model_path)
		epoch_log = ('Epoch[%d] Val-Accuracy: %.4f, Val-loss: %.4f' %
			        (cur_step, self.Val_acc, Val_epoch_loss))
		logger.info(epoch_log)
		writer.add_scalar('Accuracy/Val', self.Val_acc, cur_step)
		writer.add_scalar('Loss/Val', Val_epoch_loss, cur_step)
		if self.Val_acc > self.best_Val_acc:
			print("Saving..")
			print("best_Val_acc: %0.3f" % self.Val_acc)
			best_val_acc = ("Best_Val_acc: %0.3f" % self.Val_acc)
			logger.info(best_val_acc)
			state = {
			    'net' : net.state_dict() if use_cuda else net,
			    'best_val_acc' : self.Val_acc,
			    'best_val_epoch' : epoch,
			}
			torch.save(state, os.path.join(path, 'Val_model.t7'))
			self.best_Val_acc = self.Val_acc
			self.best_Val_acc_epoch = epoch

	def test_model(self, Testloader, criterion, net, epoch, logger, writer, path):
		#global Test_acc
		#global best_Test_acc
		#global best_Test_acc_epoch
		use_cuda = torch.cuda.is_available()
		net.eval()
		Test_loss = 0
		correct = 0
		total = 0
		for batch_idx, (inputs, targets) in enumerate(Testloader):
			if use_cuda:
				inputs, targets = inputs.cuda(device = self.device_ids[0]),\
				                  targets.cuda(device = self.device_ids[0])
			inputs, targets = Variable(inputs), Variable(targets)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			Test_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum()
		## logging.
		self.Test_acc = 100.*float(correct)/total
		Test_epoch_loss = Test_loss / (batch_idx + 1)
		epoch_log = ('Epoch[%d] Test-Accuracy: %.4f, Test-Loss: %.4f'%
			        (epoch + 1, self.Test_acc, Test_epoch_loss))
		logger.info(epoch_log)
		writer.add_scalar('Accuracy/test', self.Test_acc, epoch + 1)
		writer.add_scalar('Loss/test', Test_epoch_loss, epoch + 1)
		## save checkpoint
		if self.Test_acc > self.best_Test_acc:
			print('Saving..')
			print('best_Test_acc: %0.3f' % self.Test_acc)
			best_test_acc = ('Best_Test_Accuracy: %0.3f' % self.Test_acc)
			logger.info(best_test_acc)
			state = {
			    'net': net.state_dict() if use_cuda else net,
			    'best_Test_acc': self.best_Test_acc,
			    'best_Test_acc_epoch': epoch,
			}
			if not os.path.isdir(path):
				os.mkdir(path)
			torch.save(state, os.path.join(path, 'Test_model.t7'))
			self.best_Test_acc = self.Test_acc
			self.best_Test_acc_epoch = epoch

	def train_model(self, net, criterion, optimizer, scheduler, trainloader, valloader, testloader, logger, writer, path):
		use_cuda = torch.cuda.is_available()
		## if or not pre
		if self.pretrained:
			checkpoint = torch.load(os.path.join(self.pretrained))
			net.load_state_dict(checkpoint['net'])
			print("==> Loaded checkpoint from pretrained model-'{}'".format(self.pretrained))
		## resume_from
		elif self.resume_from:
			print('Loading weight...')
			checkpoint = torch.load(os.path.join(self.resume_from))
			net.load_state_dict(checkpoint['net'])
			acc = checkpoint['acc']
			cur_epoch = checkpoint['epoch']
			current_epoch = cur_epoch
			print("=> Loaded checkpoint='{}' (epoch={})".format(self.resume_from, current_epoch))
		else:
			resume_from = 0
			current_epoch = 0
			print('==> Building model..')

		idx_ter = 0
		for epoch in range(current_epoch, self.num_epochs):
			print('Epoch {}/{}'.format(epoch, self.num_epochs))
			global Train_acc
			net.train()
			train_loss = 0
			correct = 0
			total = 0
			current_lr = optimizer.param_groups[0]['lr']
			#warmup_steps = 500
			for batch_idx, (inputs, targets) in enumerate(trainloader):
				## warmup
				if batch_idx <= self.warmup_steps and epoch == 0:
					warmup_percent_done = (batch_idx + 1) / (self.warmup_steps + 1)
					warmup_lr = float(self.init_lr * warmup_percent_done)
					current_lr = warmup_lr
					set_lr(optimizer, current_lr)
				else:
					current_lr = current_lr
				if use_cuda:
					inputs, targets = inputs.cuda(device = self.device_ids[0]),\
					                  targets.cuda(device = self.device_ids[0])
				inputs, targets = Variable(inputs), Variable(targets)
				## mixup or cutmix
				if self.mixup or self.cutmix:
					if epoch <=self.num_epochs - 20:
						if self.mixup and (not self.cutmix):
							x_batch, y_batch_a, y_batch_b, lam = mixup_data_radio(inputs, targets, alpha = 0.25, mix_radio = 0.5)
						elif self.cutmix and (not self.mixup):
							x_batch, y_batch_a, y_batch_b, lam = cutmix_data_radio(inputs, targets, alpha = 0.25, mix_radio = 0.5)
						elif self.mixup and self.cutmix:
							x_batch, y_batch_a, y_batch_b, lam = cutmix_data_radio(inputs, targets, alpha=0.3, mix_radio=0.5)\
							if np.random.rand() > 0.5 else mixup_data_radio(inputs, targets, alpha=0.1, mix_radio=0.5)
						## save mix_img
						if self.save_mix_results:
							self.save_mix_images(x_batch = x_batch, epoch = epoch,  batch_idx = batch_idx, end_epoch = 0, end_batch_idx = 5)
						outputs = net(x_batch.cuda())
						loss = mixup_criterion(criterion, outputs, y_batch_a.cuda(device=self.device_ids[0]), y_batch_b.cuda(device=self.device_ids[0]), lam)
						optimizer.zero_grad()
						loss.backward()
						clip_gradient(optimizer, 0.1)
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
						clip_gradient(optimizer, 0.1)
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
					clip_gradient(optimizer, 0.1)
					optimizer.step()
					train_loss += loss.item()
					_, predicted = torch.max(outputs.data, 1)
					total += targets.size(0)
					correct += predicted.eq(targets.data).cpu().sum()

				accuracy = 100. * float(correct)/total
				##logging
				if batch_idx % 20 == 0:
					train_log = ('Epoch[%d/%d] Batch[%d] lr: %.6f, Training_Loss=%.4f, Train_Accuracy=%.4f' %
						(epoch + 1, self.num_epochs, batch_idx, current_lr, train_loss / (batch_idx + 1), accuracy))
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
			if not self.evel_only:
				print('Starting valing..')
				self.eval_model(valloader, criterion, net, epoch, logger, writer, path)
				print('Starting testing...')
				self.test_model(testloader, criterion, net, epoch, logger, writer, path)
			else:
				print('Only valing..')
				self.eval_model(valloader, criterion, net, epoch, logger, writer, path)

	def train(self):
		use_cuda = torch.cuda.is_available()
		path = os.path.join('./out_models/' + self.model_name + '_' + self.task_name + '_' + self.job_id)
		## get logger
		logger = self.get_logger(self.model_name, self.task_name, self.job_id, path)
		logger.info("Job_id : {}".format(self.job_id))
		logger.info("gpus_device_ids : {}".format(self.device_ids))
		logger.info("Task Name : {}".format(self.task_name))
		logger.info("Backbone_name : {}".format(self.model_name))
		logger.info("input_shape : ({},{}.{})".format(self.input_shape[0],self.input_shape[1],self.input_shape[2]))
		logger.info("batch_size : {}".format(self.batch_size))
		logger.info("num_epochs : {}".format(self.num_epochs))
		logger.info("warmup_steps : {}".format(self.warmup_steps))
		logger.info("resume_from : {}".format(self.resume_from))
		logger.info("pretrained : {}".format(self.pretrained))
		logger.info("mixup : {}".format(self.mixup))
		logger.info("cutmix : {}".format(self.cutmix))
		## tensorboard writer
		log_dir = os.path.join(path,"{}".format("tensorboard_log"))
		if not os.path.isdir(log_dir):
			os.mkdir(log_dir)
		writer = SummaryWriter(log_dir)
		## get model of train
		net = get_model(self.model_name)
		net = torch.nn.DataParallel(net, device_ids = self.device_ids)
		net = net.cuda(device = self.device_ids[0])
		## loss
		criterion = nn.CrossEntropyLoss()
		## optimizer
		if self.optimizers == 'SGD':
			optimizer = optim.SGD(net.parameters(), lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)
		elif self.optimizers == 'Adam':
			optimizer = optim.Adam(net.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
		milestones = [80,150,200,300]
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
		logger.info(("===========opti=========="))
		logger.info("Optimizer:{}".format(self.optimizers))
		logger.info("lr:{}".format(self.init_lr))
		logger.info("weight_decay:{}".format(self.weight_decay))
		logger.info("lr_scheduler: MultiStepLR")
		logger.info("milestones:{}".format(milestones))
	    ## augumation
		normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
		## train aug
		transform_train = transforms.Compose([
		    transforms.RandomCrop(int(self.input_shape[-1])),
		    transforms.RandomHorizontalFlip(),
		    transforms.RandomBrightness(brightness = self.brightness, brightness_ratio=self.brightness_ratio),
		    transforms.RandomBlur(blur_ratio = self.blur_ratio),
		    transforms.RandomRotation(degrees = self.degrees, rotation_ratio = 0.1),
		    transforms.ColorJitter(brightness = self.color_brightnesss, contrast = self.color_contrast,\
		                           saturation = self.color_saturation, hue=0),
		    transforms.ToTensor(),
		    #normalize,
		])
		## test aug
		transform_test = transforms.Compose([
		    transforms.CenterCrop(int(self.input_shape[-1])),
		    transforms.ToTensor(),
		    #normalize,
		])
		logger.info(("============aug==========="))
		logger.info("crop: RandomCrop")
		logger.info("RandomHorizontalFlip: True")
		logger.info("brightness:{}".format(self.brightness))
		logger.info("brightness_ratio:{}".format(self.brightness_ratio))
		logger.info("blur_ratio:{}".format(self.blur_ratio))
		logger.info("degrees:{}".format(self.degrees))
		logger.info("color_brightnesss:{}".format(self.color_brightnesss))
		logger.info("color_contrast:{}".format(self.color_contrast))
		logger.info("color_saturation:{}".format(self.color_saturation))
        ## prepara data
		print('==> Preparing data..')
		logger.info(("==========Datasets========="))
		logger.info("train_datasets:{}".format(self.train_datasets))
		logger.info("val_datasets:{}".format(self.val_datasets))
		logger.info("test_datasets:{}".format(self.test_datasets))
		#trainset = DataLoader(split = 'Training', transform=transform_train)
		trainset = DataLoader(self.train_datasets, self.val_datasets, self.test_datasets, split = 'Training', transform=transform_train)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size * len(self.device_ids), shuffle=True)
		Valset = DataLoader(self.train_datasets, self.val_datasets, self.test_datasets, split = 'Valing', transform=transform_test)
		Valloader = torch.utils.data.DataLoader(Valset, batch_size=64 * len(self.device_ids), shuffle=False)
		Testset = DataLoader(self.train_datasets, self.val_datasets, self.test_datasets, split = 'Testing', transform=transform_test)
		Testloader = torch.utils.data.DataLoader(Testset, batch_size=64 * len(self.device_ids), shuffle=False)
		## train
		logger.info(("======Begain Training======"))
		#self.train_model(net, criterion, optimizer, scheduler, trainloader, Valloader, Testloader, logger, writer, path)
		self.train_model(net, criterion, optimizer, scheduler, trainloader, Valloader, Testloader, logger, writer, path)
		logger.info(("======Finsh Training !!!======"))    
		logger.info(("best_val_acc_epoch: %d, best_val_acc: %0.3f" % (self.best_Val_acc_epoch, self.best_Val_acc)))
		logger.info(("best_test_acc_epoch: %d, best_test_acc: %0.3f" % (self.best_Test_acc_epoch, self.best_Test_acc)))