#iceberg v1.2
#Lenix Lobo

import numpy as np 
import pandas as pd 
import json
from subprocess import check_output

import os
import sys
import logging
#Pytorch essentials
import torch
from torch.utils.data.dataset import Dataset 
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torch import nn 
import torch.nn.functional as fu 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
#Sklearn essentials
from sklearn.metrics import roc_auc_score,log_loss,auc
from sklearn.cross_validation import StratifiedKFold,ShuffleSplit,cross_val_score,train_test_split


#use_cuda = torch.cuda.is_available()
use_cuda = False
floatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
longTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = floatTensor

lgr = logging.getLogger(__name__)
lgr.info("use cuda = "+str(use_cuda))

target_var = 'target'
base_folder = './data/processed'

#Hyper Params
batch_size = 128
global_epochs = 55
validationRatio =  0.11
learning_rate = 0.00053
momentum = 0.95
if use_cuda:
	num_workers = 0
else:
	num_workers= 4

global_seed = 999

def seed_fix(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	if use_cuda:
		torch.cuda.manual_seed(seed)

#converting X dataset to Tensors for operations
def TensorX(x):
	x = np.array(x,dtype=np.float32)
	if use_cuda:
		tensorX = (torch.from_numpy(x).cuda())
	else:
		tensorX  = (torch.from_numpy(x))

	return tensorX

def TensorY(y):
	y = y.reshape((y.shape[0],1))
	if use_cuda:
		tensorY = (torch.from_numpy(y)).type(torch.FloatTensor).cuda()
	else:
		tensorY = (torch.from_numpy(y)).type(torch.FloatTensor)

	return tensorY


#compare with original class since metaclass isnt able to compile
class Training_Dataset(Dataset):
	def __init__(self,dataset_full,offset,length):
		self.dataset_full = dataset_full
		self.offset = offset
		self.length = length
		assert len(dataset_full) >= offset + length,Exception("Parent dataset not long enough")
		super(Training_Dataset,self).__init__()

	def __len__(self):
		return self.length

	def __getitem__(self,idx):
		return self.dataset_full[idx + self.offset]

def trainTestSplit(dataset,val_share = validationRatio):
	val_offset = int(len(dataset)*(1-val_share))

	return Training_Dataset(dataset,0,val_offset),Training_Dataset(dataset,val_offset,len(dataset)-val_offset)

def read_shuffle_data(seed_num):
	seed_fix(seed_num)
	local_data = pd.read_json(base_folder+'/train.json')

	#local_data = local_data.shuffle(local_data)
	local_data = local_data.reindex(np.random.permutation(local_data.index))

	local_data['band_1'] = local_data['band_1'].apply(lambda x: np.array(x).reshape(75,75))
	local_data['band_2'] = local_data['band_2'].apply(lambda x:np.array(x).reshape(75,75))
	
	local_data['inc_angle'] = pd.to_numeric(local_data['inc_angle'],errors='coerce')
	band_1 = np.concatenate([im for im in local_data['band_1']]).reshape(-1,75,75)
	band_2 = np.concatenate([im for im in local_data['band_2']]).reshape(-1,75,75)
	local_full_img = np.stack([band_1,band_2],axis=1)
	return local_data,local_full_img

def getTrainValLoaders():
	train_imgs = TensorX(full_img)
	train_targets = TensorY(data['is_iceberg'].values)
	dset_train = TensorDataset(train_imgs,train_targets)

	local_train_ds,local_val_ds = trainTestSplit(dset_train)
	local_train_loader = torch.utils.data.DataLoader(local_train_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)
	local_val_loader = torch.utils.data.DataLoader(local_val_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)

	return local_train_loader,local_val_loader,local_train_ds,local_val_ds


def getCustomTrainValLoaders():
    # global train_ds, train_loa
	from random import randrange

	X_train,X_val,y_train,y_val = train_test_split(full_img,data['is_iceberg'].values,test_size = validationRatio,random_state=global_seed)

	local_train_ds = IceBergDataset(X_train,y_train,transform=transforms.Compose([HorizontalFlip(),VerticalFlip(),Convert_to_tensor()]))

	local_val_ds = IceBergDataset(X_val,y_val,transform=transforms.Compose([HorizontalFlip(),VerticalFlip(),Convert_to_tensor()]))

	local_train_loader = DataLoader(dataset = local_train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
	local_val_loader = DataLoader(dataset = local_val_ds,batch_size=batch_size,shuffle=True,num_workers=0)

	return local_train_loader,local_val_loader,local_train_ds,local_val_ds

#Lets build the architecture of the Actual Convolutional Network that will adapt to the dataset

#hyper params for the network to be constructed 
dropout = torch.nn.Dropout(p=0.30)
relu = torch.nn.LeakyReLU()
pool = nn.MaxPool2d(2,2)

class ConvRes(nn.Module):
	def __init__(self,insize,outsize):
		super(ConvRes,self).__init__()
		drate = .3
		self.math = nn.Sequential(nn.BatchNorm2d(insize),torch.nn.Conv2d(insize,outsize,kernel_size=2,padding=2),nn.PReLU(),)

	def forward(self,x):
		return self.math(x)

class ConvCNN(nn.Module):
	def __init__(self,insize,outsize,kernel_size=7,padding=2,pool=2,avg=True):
		super(ConvCNN,self).__init__()
		self.avg = avg
		self.math = torch.nn.Sequential(
			torch.nn.Conv2d(insize,outsize,kernel_size=kernel_size,padding=padding),
			torch.nn.BatchNorm2d(outsize),
			torch.nn.LeakyReLU(),
			torch.nn.MaxPool2d(pool,pool),
		)
		self.avgpool = torch.nn.AvgPool2d(pool,pool)

	def forward(self,x):
		x = self.math(x)
		#if self.avg is True:
		#	x = self.avgpool(x)
		return x


class SimpleNet(nn.Module):
	def __init__(self):
		super(SimpleNet,self).__init__()

		self.avgpool = nn.AdaptiveAvgPool2d(1)

		self.cnn1 = ConvCNN(2,32,kernel_size=7,pool=4,avg=False)
		self.cnn2 = ConvCNN(32,32,kernel_size=5,pool=2,avg=True)
		self.cnn3 = ConvCNN(32,32,kernel_size=5,pool=3,avg=True)

		self.res1 = ConvRes(32,64)

		self.features = nn.Sequential(
			self.cnn1,dropout,
			self.cnn2,
			self.cnn3,
			self.res1,
			)

		self.classifier = torch.nn.Sequential(
			nn.Linear(2304,1),
			)

		self.sig = nn.Sigmoid()



	def forward(self,x):
		x = self.features(x)
		x = x.view(x.size(0),-1)
		x = self.classifier(x)
		x = self.sig(x)
		return x

class ResNetLike(nn.Module):
	def __init__(self,block,layers,num_channels=2,num_classes=1):
		self.inplanes = 32
		super(ResNetLike,self).__init__()

		self.conv1 = nn.Conv2d(num_channels,32,kernel_size=7,stride=2,padding=3,bias=False)

		self.bn1 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.layer1 = self._make_layer(block,32,layers[0])
		self.dropout1 = nn.Dropout2d(p=0.3)
		self.layer2 = self._make_layer(block,64,layers[1],stride=2)
		#add similar layers to improve efficiency later
		# self.dropout2 = nn.Dropout2d(p=0.3)
		# self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
		# self.dropout3 = nn.Dropout2d(p=0.3)
		# self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7)
		self.fc = nn.Linear(64,num_classes)
		self.sig = nn.Sigmoid()

		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
				m.weight.data.normal_(0,math.sqrt(2./n))
			elif isinstance(m,nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self,block,planes,blocks,stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes*block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,bias=False),
				nn.BatchNorm2d(planes*block.expansion),
				)

		layers = []
		layers.append(block(self.inplanes,planes,stride,downsample))
		self.inplanes = planes*block.expansion
		for i in range(1,blocks):
			layers.append(block(self.inplanes,planes))

		return nn.Sequential(*layers)

	def forward(self,x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.dropout1(x)
		x = self.layer2(x)
		# x = self.dropout2(x)
		# x = self.layer3(x)
		# x = self.dropout3(x)
		# x = self.layer4(x)
		x = self.avgpool(x)
		x = x.view(x.size(0),-1)
		#print(x.data.shape)
		x = self.fc1(x)
		x = self.sig(x)

		return x

import math

def savePred(df_pred,val_score):
	csv_path = str(val_score)+'_sample_submission.csv'
	#csv_path = str(val_score)+'sumbission.csv'
	df_pred.to_csv(csv_path,columns=('id','is_iceberg'),index=None)
	print(csv_path)

def generateModel(model,num_epochs=global_epochs):
	loss_function = nn.BCELoss()
	#binary cross entropy loss

	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)#Read:L2 regularisation
	if use_cuda:
		model.cuda()
		loss_function.cuda()
	criterion = loss_function
	all_losses = []
	val_losses = []

	for epoch in range(num_epochs):
		print("{} - Epoch".format(epoch+1))

		running_loss = 0.0
		running_accuracy = 0.0

		for i,row_data in enumerate(train_loader,1):
			img,label = row_data

			if use_cuda:
				img,label = Variable(img.cuda(async=True)),Variable(label.cuda(async=True))
			else:
				img,label = Variable(img),Variable(label)


			out = model(img)
			loss = criterion(out,label)
			running_loss += loss.data[0]*label.size(0)


			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


		model.eval()
		eval_loss = 0
		eval_acc = 0
		for row_data in val_loader:
			img,label = row_data

			if use_cuda:
				img,label = Variable(img.cuda(async=True),volatile=True),Variable(label.cuda(async=True),volatile=True)
			else:
				img = Variable(img,volatile=True)
				label = Variable(label,volatile=True)
			out = model(img)
			loss = criterion(out,label)
			eval_loss += loss.data[0] * label.size(0)

		val_losses.append(eval_loss/len(val_ds))

	print("Loss:{:.6f}".format(running_loss/(len(train_ds))))
	val_result = "{:.6f}".format(eval_loss/len(val_ds))
	return model,val_result

def testModel():
	print("in test Model")
	"""
	local_data_test = pd.read_json(base_folder+'/test.json')
	#local_data = local_data.shuffle(local_data)
	local_data_test = local_data_test.reindex(np.random.permutation(local_data_test.index))

	local_data_test['band_1'] = local_data_test['band_1'].apply(lambda x: np.array(x).reshape(75,75))
	local_data_test['band_2'] = local_data_test['band_2'].apply(lambda x:np.array(x).reshape(75,75))
	
	local_data_test['inc_angle'] = pd.to_numeric(local_data_test['inc_angle'],errors='coerce')
	band_1 = np.concatenate([im for im in local_data_test['band_1']]).reshape(-1,75,75)
	band_2 = np.concatenate([im for im in local_data_test['band_2']]).reshape(-1,75,75)
	local_full_img = np.stack([band_1,band_2],axis=1)
	"""
	df_test_set = pd.read_json(base_folder+"/test.json")
	###############ignore:local_data = pd.read_json(base_folder+'/train.json')
	#df_test_set = json.loads(test_loc)
	df_test_set['band_1'] = df_test_set['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
	df_test_set['band_2'] = df_test_set['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
	
	df_test_set['inc_angle'] = pd.to_numeric(df_test_set['inc_angle'], errors='coerce')
	
	df_test_set.head(3)
	print(df_test_set.shape)
	columns = ['id', 'is_iceberg']
	df_pred = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)
	# df_pred.id.astype(int)
	for index,row in range(100),range(100):
		row_no_id = row.drop('id')
		band_1_test = (row_no_id['band_1']).reshape(-1,75,75)
		band_2_test = (row_no_id['band_2']).reshape(-1,75,75)
		full_img_test = np.stack([band_1_test,band_2_test],axis=1)

		x_data_np = np.array(full_img_test,dtype=np.float32)
		if use_cuda:
			X_tensor_test = Variable(torch.from_numpy(x_data_np).cuda())
		else:
			X_tensor_test = Variable(torch.from_numpy(x_data_np))

		predicted_val = (model(X_tensor_test).data).float()
		p_test = predicted_val.cpu().numpy().float()
		df_pred = df_pred.append({'id':row['id'],'is_iceberg':p_test},ignore_index=True)

	return df_pred



if __name__ == '__main__':
	seed_fix(global_seed)
	model = SimpleNet()
	#model = ResNetLike(BasicBlock, [1, 3, 3, 1], num_channels=2, num_classes=1)
	for i in range(0,10):
		print("Count:{}".format(i+1))
		#model = SimpleNet()

		data,full_img = read_shuffle_data(seed_num=global_seed)
		train_loader,val_loader,train_ds,val_ds = getTrainValLoaders()

		model,val_result = generateModel(model,num_epochs=75)


	df_pred=testModel()
	savePred(df_pred,val_result)