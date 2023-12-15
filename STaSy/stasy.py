# Code from https://colab.research.google.com/drive/1d_GghBJgm3qbFfaHd7EcXfEN0yf4oixj#scrollTo=-24lvCha8vz3

import numpy as np
import pandas as pd
from STaSy.models import ncsnpp_tabular
import STaSy.losses as losses
import STaSy.likelihood as likelihood
import STaSy.sampling as sampling_
from STaSy.models import utils as mutils
from STaSy.models.ema import ExponentialMovingAverage
import STaSy.datasets as datasets
from torch.utils.data import DataLoader
import STaSy.sde_lib as sde_lib
from absl import flags
import torch
from STaSy.utils import save_checkpoint, restore_checkpoint, apply_activate
import collections
from torch.utils import tensorboard
import os
from ml_collections import config_flags, config_dict

def STaSy_model(numpy_data, categorical_columns=[], ordinal_columns=[], seed=42, epochs=10000,
	activation = 'elu', layer_type = 'concatsquash', sde = 'vesde', lr = 2e-3, num_scales = 50, ngen=1, num_samples=None):

	config = config_dict.ConfigDict()
	config.workdir = "stasy"
	config.seed = seed
	config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

	if num_samples is None:
		num_samples = numpy_data.shape[0]

	config.training = training = config_dict.ConfigDict()
	training.batch_size = 1000
	training.epoch = epochs
	training.likelihood_weighting = False
	training.continuous = True
	training.reduce_mean = False
	training.eps = 1e-05
	training.loss_weighting = False
	training.spl = False # Naive Stasy to prevent problems, see https://github.com/JayoungKim408/STaSy/issues/1  
	training.lambda_ = 0.5
	training.sde = sde
	training.n_iters = 100000
	training.tolerance = 1e-03
	training.hutchinson_type = "Rademacher"
	training.retrain_type = "median"
	training.eps_iters = 1
	training.fine_tune_epochs = 0

	config.sampling = sampling = config_dict.ConfigDict()
	sampling.n_steps_each = 1
	sampling.noise_removal = False
	sampling.probability_flow = True
	sampling.snr = 0.16
	sampling.method = 'ode'
	sampling.predictor = 'euler_maruyama'
	sampling.corrector = 'none'

	config.data = data = config_dict.ConfigDict()
	data.centered = False
	data.uniform_dequantization = False

	config.model = model = config_dict.ConfigDict()
	model.nf = 64
	model.hidden_dims = (64, 128, 256, 128, 64) # made smaller to reduce problems, see https://github.com/JayoungKim408/STaSy/issues/1 
	model.conditional = True
	model.embedding_type = 'fourier'
	model.fourier_scale = 16
	model.layer_type = layer_type
	model.name = 'ncsnpp_tabular'
	model.scale_by_sigma = False
	model.ema_rate = 0.9999
	model.activation = activation
	model.sigma_min = 0.01
	model.sigma_max = 10.
	model.num_scales = num_scales
	model.alpha0 = 0.3
	model.beta0 = 0.95
	model.beta_min = 0.1
	model.beta_max = 20.0

	config.optim = optim = config_dict.ConfigDict()
	optim.weight_decay = 0
	optim.optimizer = 'Adam'
	optim.lr = lr
	optim.beta1 = 0.9
	optim.eps = 1e-8
	optim.warmup = 5000
	optim.grad_clip = 1.

	#@title Build a score mode network and dataset

	# Build data iterators
	train_ds, transformer = datasets.get_dataset(numpy_data, categorical_columns, ordinal_columns, config, uniform_dequantization=config.data.uniform_dequantization)
	data.image_size = train_ds.shape[1]
	train_iter = DataLoader(train_ds, batch_size=config.training.batch_size)

	score_model = mutils.create_model(config)
	num_params = sum(p.numel() for p in score_model.parameters())

	#@title Setup SDEs

	# Setup SDEs
	if config.training.sde.lower() == 'vpsde':
	 	sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
	 	sampling_eps = 1e-3
	elif config.training.sde.lower() == 'subvpsde':
		sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
		sampling_eps = 1e-3
	elif config.training.sde.lower() == 'vesde':
		sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
		sampling_eps = 1e-5
	else:
		raise NotImplementedError(f"SDE {config.training.sde} unknown.")

	#@title Build utilities for training
	tb_dir = os.path.join(config.workdir, "tensorboard")
	os.makedirs(tb_dir, exist_ok=True)
	writer = tensorboard.SummaryWriter(tb_dir)

	ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
	optimizer = losses.get_optimizer(config, score_model.parameters()) # Adam optimizer, lr 2e-3
	state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

	checkpoint_dir = os.path.join(config.workdir, "checkpoints")
	checkpoint_meta_dir = os.path.join(config.workdir, "checkpoints-meta", "checkpoint.pth")
	checkpoint_finetune_dir = os.path.join(config.workdir, "checkpoints_finetune")

	os.makedirs(checkpoint_dir, exist_ok=True)
	os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
	os.makedirs(checkpoint_finetune_dir, exist_ok=True)

	scaler = datasets.get_data_scaler(config)
	inverse_scaler = datasets.get_data_inverse_scaler(config)

	optimize_fn = losses.optimization_manager(config)
	continuous = config.training.continuous
	reduce_mean = config.training.reduce_mean
	likelihood_weighting = config.training.likelihood_weighting

	def loss_fn(model, batch):
		score_fn = mutils.get_score_fn(sde, model, train=True, continuous=continuous)
		t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - 1e-5) + 1e-5
		z = torch.randn_like(batch)
		mean, std = sde.marginal_prob(batch, t)
		perturbed_data = mean + std[:, None] * z

		score = score_fn(perturbed_data, t)

		loss_values = torch.square(score * std[:, None] + z)
		loss_values = torch.mean(loss_values.reshape(loss_values.shape[0], -1), dim=-1)

		return loss_values

	# Building sampling functions
	sampling_shape = (config.training.batch_size, config.data.image_size)
	sampling_fn = sampling_.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

	#@title Build utilities for v scheduling

	def min_max_scaling(factor, scale=(0, 1)):

		std = (factor - factor.min()) / (factor.max() - factor.min())
		new_min = torch.tensor(scale[0])
		new_max = torch.tensor(scale[1])
		return std * (new_max - new_min) + new_min


	def compute_v(ll, alpha, beta):

		v = -torch.ones(ll.shape).to(ll.device)
		v[torch.gt(ll, beta)] = torch.tensor(0., device=v.device)
		v[torch.le(ll, alpha)] = torch.tensor(1., device=v.device)

		if ll[torch.eq(v, -1)].shape[0] !=0 and ll[torch.eq(v, -1)].shape[0] !=1 :
					v[torch.eq(v, -1)] = min_max_scaling(ll[torch.eq(v, -1)], scale=(1, 0)).to(v.device)
		else:
					v[torch.eq(v, -1)] = torch.tensor(0.5, device=v.device)

		return v

	#@title Start model training
	alpha0 = config.model.alpha0
	beta0 = config.model.beta0

	for epoch in range(config.training.epoch+1):
		state['epoch'] += 1
		for iteration, batch in enumerate(train_iter):
			batch = batch.to(config.device).float()
			# loss = train_step_fn(state, batch)

			# model = state['model']
			optimizer = state['optimizer']
			optimizer.zero_grad()
			loss_values = loss_fn(score_model, batch)

			q_alpha = torch.tensor(alpha0 + torch.log( torch.tensor(1+ 0.0001718*state['step']* (1-alpha0), dtype=torch.float32) )).clamp_(max=1).to(loss_values.device)
			q_beta = torch.tensor(beta0 + torch.log( torch.tensor(1+ 0.0001718*state['step']* (1-beta0), dtype=torch.float32) )).clamp_(max=1).to(loss_values.device)

			alpha = torch.quantile(loss_values, q_alpha)
			beta = torch.quantile(loss_values, q_beta)
			assert alpha <= beta
			v = compute_v(loss_values, alpha, beta)
			loss = torch.mean(v*loss_values)

			loss.backward()
			optimize_fn(optimizer, score_model.parameters(), step=state['step'])
			state['step'] += 1
			state['ema'].update(score_model.parameters())

		if epoch % 500 == 0:
			print("epoch: %d, iter: %d, training_loss: %.5e, q_alpha: %.3e, q_beta: %.3e" % (epoch, iteration, loss.item(), q_alpha, q_beta))
		if epoch % 10 == 0:
			save_checkpoint(checkpoint_meta_dir, state)

	# Generate samples
	sampling_shape = (ngen*num_samples, config.data.image_size)
	samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)
	samples = apply_activate(samples, transformer.output_info)
	samples = transformer.inverse_transform(samples.cpu().numpy())

	return samples
