import os

import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch

import PIL

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *

from IPython.display import display

import image_utils

# ---------------------------- ARGS ----------------------------
parser = argparse.ArgumentParser(description='tree-ring_inversion_experiment')
parser.add_argument('--run_name', default='tree-ring_inversion_experiment')
parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--end', default=1, type=int)
parser.add_argument('--image_length', default=512, type=int)
parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')  # PNDMScheduler
parser.add_argument('--model_id_extract', default='runwayml/stable-diffusion-v1-5')  # PNDMScheduler
#model_id_alt0 = 'stabilityai/stable-diffusion-2-1-base'  # PNDMScheduler
#model_id_alt0 = 'Kandinski'  # PNDMScheduler
#model_id_alt0 = 'runwayml/stable-diffusion-v1-5'  # PNDMScheduler
#model_id_alt0 = 'prompthero/openjourney'  # PNDMScheduler, torch_dtype=torch.float16
#model_id_alt0 = 'Fictiverse/Stable_Diffusion_Microscopic_model'  # PNDMScheduler
#model_id_alt0 = 'dalle-mini/dalle-mega'  # PNDMScheduler
parser.add_argument('--with_tracking', action='store_false', default=True)
parser.add_argument('--num_images', default=1, type=int)
parser.add_argument('--guidance_scale', default=7.5, type=float)
parser.add_argument('--num_inference_steps', default=50, type=int)
parser.add_argument('--test_num_inference_steps', default=None, type=int)
parser.add_argument('--reference_model', default='ViT-g-14')
parser.add_argument('--reference_model_pretrain', default='laion2b_s12b_b42k')
parser.add_argument('--max_num_log_image', default=100, type=int)
parser.add_argument('--gen_seed', default=0, type=int)

# watermark
parser.add_argument('--w_seed', default=999999, type=int)
parser.add_argument('--w_channel', default=3, type=int)
parser.add_argument('--w_pattern', default='ring')
parser.add_argument('--w_mask_shape', default='circle')
parser.add_argument('--w_radius', default=10, type=int)
parser.add_argument('--w_measurement', default='l1_complex')
parser.add_argument('--w_injection', default='complex')
parser.add_argument('--w_pattern_const', default=0, type=float)

# for image distortion
parser.add_argument('--r_degree', default=0, type=float)
parser.add_argument('--jpeg_ratio', default=None, type=int)
parser.add_argument('--crop_scale', default=None, type=float)
parser.add_argument('--crop_ratio', default=None, type=float)
parser.add_argument('--gaussian_blur_r', default=8, type=int)
parser.add_argument('--gaussian_std', default=None, type=float)
parser.add_argument('--brightness_factor', default=None, type=float)
parser.add_argument('--rand_aug', default=0, type=int)

args, unknown = parser.parse_known_args()

if args.test_num_inference_steps is None:
    args.test_num_inference_steps = args.num_inference_steps


# ---------------------------- LOAD Model & Data ----------------------------
# load diffusion model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE={device}')

# original pipe (used for generating images and also inversion)
print('LOAD PIPE orig')
pipe_orig = InversableStableDiffusionPipeline.from_pretrained(
    args.model_id,
    scheduler=DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler'),
    #torch_dtype=torch.float16,
    torch_dtype=torch.float32,
    #revision='fp16',
    variant='fp16',  # updated after new transformers version
    )
pipe_orig = pipe_orig.to(device)

# alternative pipe (used for inversion process only)
print('LOAD PIPE alt0')
model_id_alt0 = args.model_id_extract

pipe_alt0 = InversableStableDiffusionPipeline.from_pretrained(
    model_id_alt0,
    scheduler=DPMSolverMultistepScheduler.from_pretrained(model_id_alt0, subfolder='scheduler'),
    #torch_dtype=torch.float16,
    torch_dtype=torch.float32,
    #revision='fp16',
    variant='fp16',  # updated after new transformers version
    )
pipe_alt0 = pipe_alt0.to(device)

# dataset
dataset, prompt_key = get_dataset(args)

# ground-truth patch
gt_patch = get_watermarking_pattern(pipe_orig, args, device)

results = []
clip_scores = []
clip_scores_w = []
no_w_metrics = []
w_metrics = []


# ---------------------------- RUN Settings ----------------------------
current_prompt = dataset[0][prompt_key]

known_prompt = current_prompt # assume at the detection time, the original prompt is unknown
unknown_prompt = '' # assume at the detection time, the original prompt is unknown

seed = args.gen_seed


# ---------------------------- GENERATE WO Watermark ----------------------------
# generation without watermarking
set_random_seed(seed)
init_latents_no_w = pipe_orig.get_random_latents()
outputs_no_w = pipe_orig(
    current_prompt,
    num_images_per_prompt=args.num_images,
    guidance_scale=args.guidance_scale,
    num_inference_steps=args.num_inference_steps,
    height=args.image_length,
    width=args.image_length,
    latents=init_latents_no_w,
    )
gen_no_w = outputs_no_w.images[0]


# ---------------------------- GENERATE W Watermark ----------------------------
# generation with watermarking
if init_latents_no_w is None:
    set_random_seed(seed)
    init_latents_w = pipe_orig.get_random_latents()
else:
    init_latents_w = copy.deepcopy(init_latents_no_w)

# get watermarking mask
watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

# inject watermark
init_latents_w, init_latens_w_fft = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args,
                                                     # and-mill -------------------------------------------
                                                     return_fft=True
                                                     # and-mill -------------------------------------------
                                                     )

outputs_w = pipe_orig(
    current_prompt,
    num_images_per_prompt=args.num_images,
    guidance_scale=args.guidance_scale,
    num_inference_steps=args.num_inference_steps,
    height=args.image_length,
    width=args.image_length,
    latents=init_latents_w,
    )
gen_w = outputs_w.images[0]

# ---------------------------- SAVE no_w and w ----------------------------
image_utils.save_pil(gen_no_w, 'gen_no_w.png')
image_utils.save_pil(gen_w, 'gen_w.png')

# ---------------------------- APPLY Distortion ----------------------------
gen_no_w_distorted, gen_w_distorted = image_distortion(gen_no_w, gen_w, seed, args)

# ---------------------------- APPLY Distortion ----------------------------
gen_w_initial_wchannel = init_latens_w_fft[0][args.w_channel].real.detach().cpu().numpy()

# ---------------------------- INVERSION ----------------------------
# for defining order of cells in wandb
COLUMNS = ['prompt',
           'BASE=gen_no_w',
           'FFT(wchannel)   BASE=gen_no_w   PIPE=orig   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_no_w   PIPE=orig   INV_PROMPT=unknown_prompt',
           'FFT(wchannel)   BASE=gen_no_w   PIPE=alt0   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_no_w   PIPE=alt0   INV_PROMPT=unknown_prompt',
           'BASE=gen_no_w_distorted',
           'FFT(wchannel)   BASE=gen_no_w_distorted   PIPE=orig   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_no_w_distorted   PIPE=orig   INV_PROMPT=unknown_prompt',
           'FFT(wchannel)   BASE=gen_no_w_distorted   PIPE=alt0   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_no_w_distorted   PIPE=alt0   INV_PROMPT=unknown_prompt',
           'BASE=gen_w',
           'FFT(wchannel)   BASE=gen_w   PIPE=orig   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_w   PIPE=orig   INV_PROMPT=unknown_prompt',
           'FFT(wchannel)   BASE=gen_w   PIPE=alt0   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_w   PIPE=alt0   INV_PROMPT=unknown_prompt',
           'BASE=gen_w_distorted',
           'FFT(wchannel)   BASE=gen_w_distorted   PIPE=orig   INV_PROMPT=prompt',
           'FFT(wchannel)   BASE=gen_w_distorted   PIPE=orig   INV_PROMPT=prompt',
           'FFT(wchannel)   BASE=gen_w_distorted   PIPE=alt0   INV_PROMPT=prompt',
           'FFT(wchannel)   BASE=gen_w_distorted   PIPE=alt0   INV_PROMPT=prompt']

# these are axis to iterate over during analysis
BASES = {'gen_no_w': gen_no_w,
         'gen_no_w_distorted': gen_no_w_distorted,
         'gen_w': gen_w,
         'gen_w_distorted': gen_w_distorted}
PIPES = {'orig': pipe_orig, 'alt0': pipe_alt0}
PROMPTS = {'known_prompt': known_prompt,
           'unknown_prompt': unknown_prompt}

# start collecting the row cells
row = {'prompt': current_prompt}

# base images (w/wo watermark x distorted)
for img_key, img in BASES.items():
    row[f'BASE={img_key}'] = img

    # pipes
    for pipe_key, pipe in PIPES.items():
        # text embedding
        for prompt_key, prompt in PROMPTS.items():
        
            # generate embedding for pipe and prompt
            text_embedding = pipe.get_text_embedding(prompt)

            # get latent
            latents = pipe.get_image_latents(transform_img(img).unsqueeze(0).to(text_embedding.dtype).to(device),
                                             sample=False)
            # revert latents back to init
            extracted_init = pipe.forward_diffusion(
                latents=latents,
                text_embeddings=text_embedding,
                guidance_scale=1,
                num_inference_steps=args.test_num_inference_steps,
            )
            # fft
            extracted_init_fft_wchannel = torch.fft.fftshift(torch.fft.fft2(extracted_init), dim=(-1, -2))[0][args.w_channel].real.detach().cpu().numpy()
            # add to row
            row[f'FFT(wchannel)   BASE={img_key}   PIPE={pipe_key}   INV_PROMPT={prompt_key}'] = extracted_init_fft_wchannel


# ---------------------------- sim between ffts ----------------------------
f0 = row[f'FFT(wchannel)   BASE=gen_w   PIPE=orig   INV_PROMPT=unknown_prompt']
f1 = row[f'FFT(wchannel)   BASE=gen_w   PIPE=alt0   INV_PROMPT=unknown_prompt']
print('####################################')
print('####################################')
print('####################################')
print((f0 - f1).sum())

print('####################################')
print('####################################')
print('####################################')

# ---------------------------- SAVE WANDB ----------------------------
if args.with_tracking:

    def wandb_type(cell):
        """Helper funktion to convert data to wandb compatible format"""
        if isinstance(cell, torch.Tensor) or isinstance(cell, np.ndarray) or isinstance(cell, PIL.Image.Image):
            return wandb.Image(cell)
        elif isinstance(cell, str):
            return cell
        else:
            raise Exception(f'Unknown type: {type(cell)}')

    wandb.init(project='tree-ring_inversion_experiment')
    wandb.config.update(args)
    
    # start table
    #columns = [c for c in COLUMNS if c in row.keys()]  # only columns we actually collected
    #table = wandb.Table(columns=columns)
    #table.add_data(*[wandb_type(row[col]) for col in table.columns if col in row.keys()])
    columns = list(row.keys())
    table = wandb.Table(columns=columns)
    table.add_data(*[wandb_type(row[col]) for col in row.keys()])

    wandb.log({'table': table})

    wandb.finish()