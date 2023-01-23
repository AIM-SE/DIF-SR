from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size()[0], -1)


class model(nn.Module):
	def __init__(self, user_size, item_size, conv_channel=[64,64], conv_size=[1,3], embed_size=128, dropout=0.2, k=9, device="cuda"):
		super(model, self).__init__()
		self.user_size = user_size
		self.item_size = item_size
		self.embed_size = embed_size
		self.dropout = dropout
		self.k = k
		self.conv_channel = conv_channel
		self.conv_size = conv_size
		self.device = device

		self.embed_user_mean = nn.Linear(self.user_size, self.embed_size)
		self.embed_user_var = nn.Linear(self.user_size, self.embed_size)
		self.embed_item_mean = nn.Linear(self.item_size, self.embed_size)
		self.embed_item_var = nn.Linear(self.item_size, self.embed_size)

		self.conv = nn.Sequential(
			nn.Conv2d(self.embed_size, conv_channel[0], conv_size[0]),
			nn.ELU()
		)

		for idx, (channel, kernel_size) in enumerate(zip(conv_channel[1:], conv_size[1:])):
			self.conv.add_module("conv", nn.Conv2d(conv_channel[idx], channel, kernel_size))
			self.conv.add_module("elu", nn.ELU())


		self.fc = nn.Sequential(
			nn.Linear(conv_channel[-1] * (k - sum(conv_size) + len(conv_size)) ** 2, 512),
			nn.BatchNorm1d(512),
			nn.ELU(),
			nn.Dropout(p=self.dropout),
			nn.Linear(512, 64),
			nn.BatchNorm1d(64),
			nn.ELU(),
			nn.Dropout(p=self.dropout),
			nn.Linear(64, 1),
		)

	def convert_one_hot(self, feature, size):
		""" Convert user and item ids into one-hot format. """
		batch_size = feature.shape[0]
		feature = feature.view(batch_size, 1)
		f_onehot = torch.FloatTensor(batch_size, size).to(self.device)
		f_onehot.zero_()
		f_onehot.scatter_(-1, feature, 1)

		return f_onehot


	def forward(self, user, item):
		size = int(user.shape[0])
		user = self.convert_one_hot(user, self.user_size)
		item = self.convert_one_hot(item, self.item_size)

		user_mean = self.embed_user_mean(user).unsqueeze(1)		# (batch_size, 1, embed_size)
		user_std = (F.elu(self.embed_user_var(user)) + 1).unsqueeze(1)	# (batch_size, 1, embed_size)
		item_mean = self.embed_item_mean(item).unsqueeze(1)
		item_std = (F.elu(self.embed_item_var(item)) + 1).unsqueeze(1)

		# reparameterize tricks: eps*std + mu
		samples_user = torch.randn((size, self.k, self.embed_size)).to(self.device) * user_std + user_mean 	# (batch_size, k, embed_size)
		samples_item = torch.randn((size, self.k, self.embed_size)).to(self.device) * item_std + item_mean

		samples_user = samples_user.unsqueeze(2)			# (batch_size, k, 1, embed_size)
		samples_user = samples_user.repeat(1, 1, self.k, 1)		# (batch_size, k, k, embed_size)

		samples_item = samples_item.unsqueeze(2)
		samples_item = samples_item.repeat(1, 1, self.k, 1)		# (batch_size, k, k, embed_size)
		map = samples_user *  samples_item.transpose(1, 2)		# (batch_size, k, k, embed_size)

		x = self.conv(map.permute(0, 3, 1, 2))
		x = x.view(size, -1)
		x = self.fc(x)

		prediction = torch.sigmoid(x)

		return prediction.view(-1)

