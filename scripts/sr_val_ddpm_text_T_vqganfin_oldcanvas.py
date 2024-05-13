"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import intel_extension_for_pytorch as ipex

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
import torch.nn.functional as F
import cv2
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

def space_timesteps(num_timesteps, section_counts):
	"""
	Create a list of timesteps to use from an original diffusion process,
	given the number of timesteps we want to take from equally-sized portions
	of the original process.
	For example, if there's 300 timesteps and the section counts are [10,15,20]
	then the first 100 timesteps are strided to be 10 timesteps, the second 100
	are strided to be 15 timesteps, and the final 100 are strided to be 20.
	If the stride is a string starting with "ddim", then the fixed striding
	from the DDIM paper is used, and only one section is allowed.
	:param num_timesteps: the number of diffusion steps in the original
						  process to divide up.
	:param section_counts: either a list of numbers, or a string containing
						   comma-separated numbers, indicating the step count
						   per section. As a special case, use "ddimN" where N
						   is a number of steps to use the striding from the
						   DDIM paper.
	:return: a set of diffusion steps from the original process to use.
	"""
	if isinstance(section_counts, str):
		if section_counts.startswith("ddim"):
			desired_count = int(section_counts[len("ddim"):])
			for i in range(1, num_timesteps):
				if len(range(0, num_timesteps, i)) == desired_count:
					return set(range(0, num_timesteps, i))
			raise ValueError(
				f"cannot create exactly {num_timesteps} steps with an integer stride"
			)
		section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
	size_per = num_timesteps // len(section_counts)
	extra = num_timesteps % len(section_counts)
	start_idx = 0
	all_steps = []
	for i, section_count in enumerate(section_counts):
		size = size_per + (1 if i < extra else 0)
		if size < section_count:
			raise ValueError(
				f"cannot divide section of {size} steps into {section_count}"
			)
		if section_count <= 1:
			frac_stride = 1
		else:
			frac_stride = (size - 1) / (section_count - 1)
		cur_idx = 0.0
		taken_steps = []
		for _ in range(section_count):
			taken_steps.append(start_idx + round(cur_idx))
			cur_idx += frac_stride
		all_steps += taken_steps
		start_idx += size
	return set(all_steps)

def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model
	model.eval()
	return model

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.

def process_pictures(model, vq_model, img_list, init_image_list, opt, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, outpath = None):
	perf_time = []
	for n in trange(len(init_image_list), desc="Sampling"):
		t0 = time.time()
		init_image = init_image_list[n]
		init_image = init_image.clamp(-1.0, 1.0)
		ori_size = None

		if init_image.size(-1) < opt.input_size or init_image.size(-2) < opt.input_size:
			ori_size = init_image.size()
			new_h = max(ori_size[-2], opt.input_size)
			new_w = max(ori_size[-1], opt.input_size)
			init_template = torch.zeros(1, init_image.size(1), new_h, new_w).to(init_image.device)
			init_template[:, :, :ori_size[-2], :ori_size[-1]] = init_image
		else:
			init_template = init_image
		t1 = time.time()
		init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_template))  
		t2 = time.time()
        # move to latent space
		text_init = ['']*opt.n_samples
		semantic_c = model.cond_stage_model(text_init)
		t3 = time.time()
		noise = torch.randn_like(init_latent)
		# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
		t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
		t = t.long()
		x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
		t4 = time.time()
		samples, _ = model.sample_canvas(cond=semantic_c, struct_cond=init_latent, batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True, tile_size=int(opt.input_size/8), tile_overlap=opt.tile_overlap, batch_size_sample=opt.n_samples)
		t5 = time.time()
		_, enc_fea_lq = vq_model.encode(init_template)
		x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
		t6 = time.time()
		if ori_size is not None:
			x_samples = x_samples[:, :, :ori_size[-2], :ori_size[-1]]
		if opt.colorfix_type == 'adain':
			x_samples = adaptive_instance_normalization(x_samples, init_image)
		elif opt.colorfix_type == 'wavelet':
			x_samples = wavelet_reconstruction(x_samples, init_image)
		x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
		t7 = time.time()
		if outpath != None :
			for i in range(init_image.size(0)):
				img_name = img_list.pop(0)
				basename = os.path.splitext(os.path.basename(img_name))[0]
				x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
				Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(outpath, basename+'_hq.png'))
				init_image = torch.clamp((init_image + 1.0) / 2.0, min=0.0, max=1.0)
				init_image = 255. * rearrange(init_image[i].cpu().numpy(), 'c h w -> h w c')
				Image.fromarray(init_image.astype(np.uint8)).save(os.path.join(outpath, basename+'_lq.png'))
		perf_time.append([t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6])
	return perf_time

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--init-img",
		type=str,
		nargs="?",
		help="path to the input image",
		default="inputs/user_upload"
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="outputs/user_upload"
	)
	parser.add_argument(
		"--ddpm_steps",
		type=int,
		default=1000,
		help="number of ddpm sampling steps",
	)
	parser.add_argument(
		"--C",
		type=int,
		default=4,
		help="latent channels",
	)
	parser.add_argument(
		"--f",
		type=int,
		default=8,
		help="downsampling factor, most often 8 or 16",
	)
	parser.add_argument(
		"--n_samples",
		type=int,
		default=2,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="configs/stableSRNew/v2-finetune_text_T_512.yaml",
		help="path to config which constructs model",
	)
	parser.add_argument(
		"--ckpt",
		type=str,
		default="./stablesr_000117.ckpt",
		help="path to checkpoint of model",
	)
	parser.add_argument(
		"--vqgan_ckpt",
		type=str,
		default="./vqgan_cfw_00011.ckpt",
		help="path to checkpoint of VQGAN model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument(
		"--dec_w",
		type=float,
		default=0.5,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--tile_overlap",
		type=int,
		default=32,
		help="tile overlap size",
	)
	parser.add_argument(
		"--upscale",
		type=float,
		default=2.0,
		help="upsample scale",
	)
	parser.add_argument(
		"--colorfix_type",
		type=str,
		default="nofix",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)
	parser.add_argument(
		"--bf16",
		action='store_true',
		default=False,
		help="enable BF16",
	)
	parser.add_argument(
		"--ipex1",
		action='store_true',
		default=False,
		help="enable ipex",
	)
	parser.add_argument(
		"--ipex2",
		action='store_true',
		default=False,
		help="enable ipex",
	)
	parser.add_argument(
		"--loop",
		type=int,
		default=5,
		help="loop numbers",
	)
	parser.add_argument(
		"--profile",
		action='store_true',
		default=False,
		help="enable profiler",
	)
	opt = parser.parse_args()
	device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
	seed_everything(opt.seed)

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')



	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir

	batch_size = opt.n_samples

	img_list_ori = os.listdir(opt.init_img)
	img_list = copy.deepcopy(img_list_ori)
	init_image_list = []
	for item in img_list_ori:
		# if os.path.exists(os.path.join(outpath, item)):
		# 	img_list.remove(item)
		# 	continue
		cur_image = load_img(os.path.join(opt.init_img, item)).to(device)
		# max size: 1800 x 1800 for V100
		cur_image = F.interpolate(
				cur_image,
				size=(int(cur_image.size(-2)*opt.upscale),
					  int(cur_image.size(-1)*opt.upscale)),
				mode='bicubic',
				)
		init_image_list.append(cur_image)

	vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vqgan_ckpt)
	vq_model = vq_model.eval()

	if opt.bf16 == True:
		vq_model = vq_model.to(torch.bfloat16)

	if opt.ipex1 == True:
		vq_model = ipex.optimize(vq_model, dtype=torch.bfloat16 if opt.bf16 else torch.float32, weights_prepack=True)
	vq_model.decoder.fusion_w = opt.dec_w

	config = OmegaConf.load(f"{opt.config}")
	config.model.params.openvino_config.params.num_streams = opt.n_samples
	model = load_model_from_config(config, f"{opt.ckpt}")
	model = model.to(device)
	model.configs = config
	model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
						  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	model.num_timesteps = 1000
	sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
	sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

	use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
	last_alpha_cumprod = 1.0
	new_betas = []
	timestep_map = []
	for i, alpha_cumprod in enumerate(model.alphas_cumprod):
		if i in use_timesteps:
			new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
			last_alpha_cumprod = alpha_cumprod
			timestep_map.append(i)
	new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
	model.num_timesteps = 1000
	model.ori_timesteps = list(use_timesteps)
	model.ori_timesteps.sort()
	model = model.to(device)
	model = model.eval()
	if opt.bf16 == True:
		model = model.to(torch.bfloat16)

	if opt.ipex2 == True:
		with torch.cpu.amp.autocast(enabled=opt.bf16), torch.no_grad():        
			x = torch.randn([1, 4, 64, 64]).float()
			t = torch.ones(1, dtype=torch.int32)
			context = torch.randn([1, 77, 1024]).float()
			struct_cond = {}
			struct_cond[str(64)] = torch.randn([1, 256, 64, 64]).float()
			struct_cond[str(32)] = torch.randn([1, 256, 32, 32]).float()
			struct_cond[str(16)] = torch.randn([1, 256, 16, 16]).float()
			struct_cond[str(8)] =  torch.randn([1, 256, 8, 8]).float()
			unet_ipex = torch.jit.trace(model.model.diffusion_model, (x, t, context, struct_cond), check_trace=False, strict=False)
			model.model.diffusion_model = torch.jit.freeze(unet_ipex)

    #times in perf_time: 
    # 0: init_time, 
    # 1: first_stage_infer_time, 
    # 2: cond_stage_infer_time, 
    # 3: prepare_data_time, 
    # 4: sample_canvas_time
    # 5: vqmodel_time
    # 6: colorfix_time
	with torch.no_grad(), torch.cpu.amp.autocast(enabled=opt.bf16, dtype=torch.bfloat16):
		print(f"img size={len(init_image_list)}, loop={opt.loop}, shape={init_image_list[0].size()}")
		###warm up
		perf_time = process_pictures(model, vq_model, img_list, init_image_list, opt, 
                               sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, outpath)
		print(f"{perf_time[0]}")
        ### benchmark
		all_perf_time = []
		tic = time.time()
		for i in range(opt.loop) :
			all_perf_time.append(process_pictures(model, vq_model, img_list, init_image_list, opt, 
                                         sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod))
		toc = time.time()
		first_stage = 0.0
		cond_stage = 0.0
		sample_canvas = 0.0
		vqmodel = 0.0
		colorfix = 0.0
		for its in all_perf_time:
			for it in its:
				first_stage += it[1]
				cond_stage += it[2]
				sample_canvas += it[4]
				vqmodel += it[5]
				colorfix += it[6]
		total_size = opt.loop * len(init_image_list)
		first_stage = first_stage / total_size
		cond_stage = cond_stage / total_size
		sample_canvas = sample_canvas / total_size
		vqmodel = vqmodel / total_size
		colorfix = colorfix / total_size
		# print(f"image {total_size}, total={(toc-tic)/opt.loop}, first_stage={first_stage}, cond_stage={cond_stage}, sample_canvas={sample_canvas}, vqmodel={vqmodel}, colorfix={colorfix}")
		print("##### total time {0:8.4f} s, first_stage={1:8.4f}, cond_stage={2:8.4f}, sample_canvas={3:8.4f}, colorfix={4:8.4f}".format((toc - tic) / opt.loop, first_stage, cond_stage, sample_canvas, vqmodel, colorfix ))

if __name__ == "__main__":
	main()
