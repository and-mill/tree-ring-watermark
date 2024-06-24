import argparse
import os

import copy

import numpy as np
import torch

from diffusers import StableDiffusion3Pipeline

import matplotlib.pyplot as plt

from PIL import Image

from scipy.stats import ks_2samp


import optim_utils


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

DIR = 'latent_analysis/sd3/'
os.makedirs(DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='tree-ring_inversion_experiment')
parser.add_argument('--no_use_cache', action='store_false', dest='use_cache')
parser.add_argument('--image_length', default=512, type=int)
parser.add_argument('--w_mask_shape', default='circle')
parser.add_argument('--w_pattern', default='ring')
parser.add_argument('--w_radius', default=10, type=int)
parser.add_argument('--w_channel', default=3, type=int)
parser.add_argument('--w_seed', default=999999, type=int)
parser.add_argument('--w_injection', default='complex')
parser.add_argument('--output_type', default='pil')  # 'pil', 'latents', None
parser.add_argument('--num_inference_steps', default=50, type=int)
args, unknown = parser.parse_known_args()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

if args.use_cache and os.path.isfile(os.path.join(DIR, 'latent_w.pt')) and os.path.isfile(os.path.join(DIR, 'latent_no_w.pt')):
    init_latents_no_w = torch.load(os.path.join(DIR, 'latent_no_w.pt'))
    init_latents_w = torch.load(os.path.join(DIR, 'latent_w.pt'))
    pipe = None
else:

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                #torch_dtype=torch.float32,
                                                torch_dtype=torch.float16,
                                                )
    pipe = pipe.to(device)

    height = args.image_length or pipe.transformer.config.sample_size * pipe.vae_scale_factor
    width = args.image_length or pipe.transformer.config.sample_size * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.out_channels

    # get latent
    init_latents_no_w = pipe.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        pipe.text_encoder.dtype,
        device,
        generator=None,
        latents=None,
        ).to(device)
    
    # get watermarking mask
    init_latents_w = copy.deepcopy(init_latents_no_w)
    init_latents_w = optim_utils.inject_watermark(init_latents_w,
                                                  watermarking_mask = optim_utils.get_watermarking_mask(init_latents_w, args, device),
                                                  gt_patch=copy.deepcopy(optim_utils.get_watermarking_pattern(pipe, args, device, shape=init_latents_w.shape)).to(device),
                                                  args=args,
                                                  return_fft=False)
    
    # save pt
    torch.save(init_latents_no_w,
               os.path.join(DIR, 'latent_no_w.pt'))
    torch.save(init_latents_w,
               os.path.join(DIR, 'latent_w.pt'))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# necessary is pipe is loaded with float 16
init_latents_no_w = copy.deepcopy(init_latents_no_w).float()
init_latents_w = copy.deepcopy(init_latents_w).float()

init_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_no_w), dim=(-1, -2))[0][:].real.detach().cpu().numpy()
init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))[0][:].real.detach().cpu().numpy()

init_latents_no_w = init_latents_no_w[0].detach().cpu().numpy()
init_latents_w = init_latents_w[0].detach().cpu().numpy()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

from scipy.stats import ks_2samp

# Perform Kolmogorov-Smirnov test
stat, p_value = ks_2samp(init_latents_no_w.flatten(),
                         init_latents_w.flatten())
print(f'KS Statistic: {stat}, P-value: {p_value}')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# save latent vectors as images as much as possible
for name, lat in {'init_latents_no_w.png': init_latents_no_w,
                  'init_latents_w.png': init_latents_w,
                  'init_latents_no_w_fft.png': init_latents_no_w_fft,
                  'init_latents_w_fft.png': init_latents_w_fft}.items():
    
    # Transpose the array to channels last
    l = np.transpose(lat, (1, 2, 0))
    l = ((l - l.min()) / (l.max() - l.min()) * 255).astype(np.uint8)

    # Create an Image object
    if init_latents_no_w.shape[0] == 16:

        # save img even thoug we wont get all the channels
        image = Image.fromarray(l, mode='RGBA')
        image.save(os.path.join(DIR, name), mode='RGBA')

        # get mean and save as greyscale image
        l_mean = lat.mean(axis=0)
        l_mean = ((l_mean - l_mean.min()) / (l_mean.max() - l_mean.min()) * 255).astype(np.uint8)
        image = Image.fromarray(l_mean)
        image.save(os.path.join(DIR, name.split('.')[0] + f'_mean.png'))
        
        # for FFT, save channels seperately
        if 'fft' in name:
            for i in range(16):
                channel = Image.fromarray(l[:, :, i], mode='L')
                channel.save(os.path.join(DIR, name.split('.')[0] + f'_{i}.png'))
    else:
        # save image
        #image.save(os.path.join(DIR, name))
        raise NotImplementedError

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# save distribution
plt.hist(init_latents_no_w.flatten(),
         bins=100,  # Increase the number of bins for finer granularity
         alpha=0.5,  # Set transparency
         label='no w',
         color='blue')  # Optional: specify color
plt.hist(init_latents_w.flatten(),
         bins=100,  # Increase the number of bins for finer granularity
         alpha=0.5,  # Set transparency
         label='w',
         color='red')  # Optional: specify color
plt.legend()
plt.savefig(os.path.join(DIR, 'latent_distributions.png'))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

if pipe is None:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                    torch_dtype=torch.float16,)
    pipe = pipe.to(device)

# generate image
image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    guidance_scale=7.5,
    #height=args.image_length,
    #width=args.image_length,
    latents=torch.from_numpy(init_latents_no_w).unsqueeze(0),
    output_type='pil',
    num_inference_steps=args.num_inference_steps
).images[0]
image.save(os.path.join(DIR, f'no_wm_out_{args.num_inference_steps}.png'))

# generated WM image
image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    guidance_scale=7.5,
    #height=args.image_length,
    #width=args.image_length,
    latents=torch.from_numpy(init_latents_w).unsqueeze(0),
    output_type='pil',
    num_inference_steps=args.num_inference_steps
).images[0]
image.save(os.path.join(DIR, f'wm_out_{args.num_inference_steps}.png'))
