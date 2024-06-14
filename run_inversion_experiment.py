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
from diffusers import DPMSolverMultistepScheduler, KandinskyPipeline, KandinskyPriorPipeline
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
parser.add_argument('--model_id_alt', default='runwayml/stable-diffusion-v1-5')  # PNDMScheduler
#'stabilityai/stable-diffusion-2-1-base'  # PNDMScheduler
#'Kandinski'  # PNDMScheduler
#'runwayml/stable-diffusion-v1-5'  # PNDMScheduler
#'prompthero/openjourney'  # PNDMScheduler, torch_dtype=torch.float16
#'Fictiverse/Stable_Diffusion_Microscopic_model'  # PNDMScheduler
#'dalle-mini/dalle-mega'  # PNDMScheduler
#'hakurei/waifu-diffusion"
parser.add_argument('--with_tracking', action='store_false', default=True)
parser.add_argument('--num_images', default=1, type=int)
parser.add_argument('--guidance_scale', default=7.5, type=float)
parser.add_argument('--num_inference_steps', default=50, type=int)
parser.add_argument('--test_num_inference_steps', default=None, type=int)
parser.add_argument('--reference_model', default='ViT-g-14')
parser.add_argument('--reference_model_pretrain', default='laion2b_s12b_b42k')
parser.add_argument('--max_num_log_image', default=100, type=int)
parser.add_argument('--gen_seed', default=0, type=int)

# new
parser.add_argument('--prompt_id_in_dataset', default=0, type=int)
parser.add_argument('--swap_pipes', action='store_true', default=False)

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

if args.swap_pipes:
    print("Swapping pipes")
    otherpipe = args.model_id
    args.model_id = args.model_id_alt
    args.model_id_alt = otherpipe


# ---------------------------- DATA + RUN Settings ----------------------------
# dataset
dataset, prompt_key = get_dataset(args)

current_prompt = dataset[args.prompt_id_in_dataset][prompt_key]

known_prompt = current_prompt # assume at the detection time, the original prompt is unknown
unknown_prompt = '' # assume at the detection time, the original prompt is unknown

seed = args.gen_seed

#current_prompt = "masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, watercolor, night, turtleneck"


# ---------------------------- LOAD Model ----------------------------
# load diffusion model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE={device}')

# original pipe (used for generating images and also inversion)
print(f'LOAD PIPE orig: {args.model_id}')

if 'KANDINSKY' in str(args.model_id).upper():
    # need original stable diffusion pipe to be sure to get good latents
    pipe_for_getting_latents = InversableStableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base',
        scheduler=DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler'),
        #torch_dtype=torch.float16,
        torch_dtype=torch.float32,
        #revision='fp16',
        variant='fp16',  # updated after new transformers version
        safety_checker=None, requires_safety_checker=False,
        )
    init_latents_no_w = copy.deepcopy(pipe_for_getting_latents.get_random_latents()).to(device)
    gt_patch = copy.deepcopy(get_watermarking_pattern(pipe_for_getting_latents, args, device)).to(device)
    del pipe_for_getting_latents
    
    # load pipe
    pipe_prior = KandinskyPriorPipeline.from_pretrained('kandinsky-community/kandinsky-2-1-prior')
    pipe_prior.to("cuda")
    out = pipe_prior(current_prompt)
    image_emb = out.image_embeds
    negative_image_emb = out.negative_image_embeds
    pipe_orig = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1",
                                                  torch_dtype=torch.float32,
                                                  variant='fp16')
    pipe_orig.to("cuda")
else:
    pipe_orig = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler'),
        #torch_dtype=torch.float16,
        torch_dtype=torch.float32,
        #revision='fp16',
        variant='fp16',  # updated after new transformers version
        safety_checker=None, requires_safety_checker=False,
        )
    pipe_orig = pipe_orig.to(device)

# alternative pipe (used for inversion process only)
print(f'LOAD PIPE alt: {args.model_id_alt}')

pipe_alt = InversableStableDiffusionPipeline.from_pretrained(
    args.model_id_alt,
    scheduler=DPMSolverMultistepScheduler.from_pretrained(args.model_id_alt, subfolder='scheduler'),
    #torch_dtype=torch.float16,
    torch_dtype=torch.float32,
    #revision='fp16',
    variant='fp16',  # updated after new transformers version
    safety_checker=None, requires_safety_checker=False,
    )
pipe_alt = pipe_alt.to(device)

# ground-truth patch
if gt_patch is None:
    gt_patch = get_watermarking_pattern(pipe_orig, args, device)

results = []
clip_scores = []
clip_scores_w = []
no_w_metrics = []
w_metrics = []


# ---------------------------- GENERATE WO Watermark ----------------------------
# generation without watermarking
set_random_seed(seed)
if init_latents_no_w is None:
    init_latents_no_w = pipe_orig.get_random_latents()

if 'KANDINSKY' in str(args.model_id).upper():
    outputs_no_w = pipe_orig(
        current_prompt,
        image_embeds=image_emb,
        negative_image_embeds=negative_image_emb,
        height=args.image_length,
        width=args.image_length,
        latents=init_latents_no_w,
        num_inference_steps=args.num_inference_steps,
    )
else:
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

if 'KANDINSKY' in str(args.model_id).upper():
    outputs_w = pipe_orig(
        current_prompt,
        image_embeds=image_emb,
        negative_image_embeds=negative_image_emb,
        height=args.image_length,
        width=args.image_length,
        latents=init_latents_w,
        num_inference_steps=args.num_inference_steps,
    )
else:
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
           'gen_w_initial_wchannel',
           'BASE=gen_no_w',
           'FFT(wchannel)   BASE=gen_no_w   PIPE=orig   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_no_w   PIPE=orig   INV_PROMPT=unknown_prompt',
           'FFT(wchannel)   BASE=gen_no_w   PIPE=alt   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_no_w   PIPE=alt   INV_PROMPT=unknown_prompt',
           'BASE=gen_no_w_distorted',
           'FFT(wchannel)   BASE=gen_no_w_distorted   PIPE=orig   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_no_w_distorted   PIPE=orig   INV_PROMPT=unknown_prompt',
           'FFT(wchannel)   BASE=gen_no_w_distorted   PIPE=alt   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_no_w_distorted   PIPE=alt   INV_PROMPT=unknown_prompt',
           'BASE=gen_w',
           'FFT(wchannel)   BASE=gen_w   PIPE=orig   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_w   PIPE=orig   INV_PROMPT=unknown_prompt',
           'FFT(wchannel)   BASE=gen_w   PIPE=alt   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_w   PIPE=alt   INV_PROMPT=unknown_prompt',
           'BASE=gen_w_distorted',
           'FFT(wchannel)   BASE=gen_w_distorted   PIPE=orig   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_w_distorted   PIPE=orig   INV_PROMPT=unknown_prompt',
           'FFT(wchannel)   BASE=gen_w_distorted   PIPE=alt   INV_PROMPT=known_prompt',
           'FFT(wchannel)   BASE=gen_w_distorted   PIPE=alt   INV_PROMPT=unknown_prompt']

# these are axis to iterate over during analysis
BASES = {'gen_no_w': gen_no_w,
         'gen_no_w_distorted': gen_no_w_distorted,
         'gen_w': gen_w,
         'gen_w_distorted': gen_w_distorted}
PIPES = {'orig': pipe_orig, 'alt': pipe_alt}
PROMPTS = {'known_prompt': known_prompt,
           'unknown_prompt': unknown_prompt}

# start collecting the row cells
row = {'prompt': current_prompt}
row['model_id'] = args.model_id
row['model_id_alt'] = args.model_id_alt

# base images (w/wo watermark x distorted)
all_ffts = []
for img_key, img in BASES.items():
    row[f'BASE={img_key}'] = img

    # pipes
    for pipe_key, pipe in PIPES.items():
        # text embedding
        for prompt_key, prompt in PROMPTS.items():
        
            if 'KANDINSKY' in str(args.model_id).upper():
                pipe = pipe_alt

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
            # save ffts seperately for alter
            all_ffts.append(extracted_init_fft_wchannel)


# ---------------------------- DIFF between ffts ----------------------------
# get min and max of context
all_ffts_np =  np.stack(all_ffts)
def scale(x, min=0, max=1):
    d = max - min
    if d == 0:
        return x * 0
    return (x - min) / (max - min)

# add initial WM fft noise
row['gen_w_initial_wchannel'] = gen_w_initial_wchannel

diffs_fft = []
# difference between initial noise and recovered noise (orig model)
fft0 = gen_w_initial_wchannel
fft1 = row[f'FFT(wchannel)   BASE=gen_w   PIPE=orig   INV_PROMPT=unknown_prompt']
diff_gen_w_orig_initial_noise = fft0 - fft1
diffs_fft.append(diff_gen_w_orig_initial_noise)
row['diff-scaled-to-all-ffts   BASE=gen_w   PIPE=orig   vary=initial_noise'] = scale(diff_gen_w_orig_initial_noise, all_ffts_np.min(), all_ffts_np.max())
# difference between initial noise and recovered noise (orig model)
fft0 = gen_w_initial_wchannel
fft1 = row[f'FFT(wchannel)   BASE=gen_w   PIPE=alt   INV_PROMPT=unknown_prompt']
diff_gen_w_alt_initial_noise = fft0 - fft1
diffs_fft.append(diff_gen_w_alt_initial_noise)
row['diff-scaled-to-all-ffts   BASE=gen_w   PIPE=alt   vary=initial_noise'] = scale(diff_gen_w_alt_initial_noise, all_ffts_np.min(), all_ffts_np.max())
# difference when using different pipes on no WM
fft0 = row[f'FFT(wchannel)   BASE=gen_no_w   PIPE=orig   INV_PROMPT=unknown_prompt']
fft1 = row[f'FFT(wchannel)   BASE=gen_no_w   PIPE=alt   INV_PROMPT=unknown_prompt']
diff_gen_no_w_pipe = fft0 - fft1
diffs_fft.append(diff_gen_no_w_pipe)
row['diff-scaled-to-all-ffts   BASE=gen_no_w   vary=pipe'] = scale(diff_gen_no_w_pipe, all_ffts_np.min(), all_ffts_np.max())
# difference when using different pipes on WM
fft0 = row[f'FFT(wchannel)   BASE=gen_w   PIPE=orig   INV_PROMPT=unknown_prompt']
fft1 = row[f'FFT(wchannel)   BASE=gen_w   PIPE=alt   INV_PROMPT=unknown_prompt']
diff_gen_w_pipe = fft0 - fft1
diffs_fft.append(diff_gen_w_pipe)
row['diff-scaled-to-all-ffts   BASE=gen_w   vary=pipe'] = scale(diff_gen_w_pipe, all_ffts_np.min(), all_ffts_np.max())
# difference when using different prompts on WM
fft0 = row[f'FFT(wchannel)   BASE=gen_w   PIPE=orig   INV_PROMPT=known_prompt']
fft1 = row[f'FFT(wchannel)   BASE=gen_w   PIPE=orig   INV_PROMPT=unknown_prompt']
diff_gen_w_prompt = fft0 - fft1
diffs_fft.append(diff_gen_w_prompt)
row['diff-scaled-to-all-ffts   BASE=gen_w   vary=prompt'] = scale(diff_gen_w_prompt, all_ffts_np.min(), all_ffts_np.max())

# scale again, only betweend diffs min max
diffs_fft_np = np.stack(diffs_fft)
row['diff-scaled-to-diff-ffts   BASE=gen_w   PIPE=orig   vary=pipe'] = scale(diff_gen_w_orig_initial_noise, diffs_fft_np.min(), diffs_fft_np.max())
row['diff-scaled-to-diff-ffts   BASE=gen_w   PIPE=alt   vary=pipe'] = scale(diff_gen_w_alt_initial_noise, diffs_fft_np.min(), diffs_fft_np.max())
row['diff-scaled-to-diff-ffts   BASE=gen_no_w   vary=pipe'] = scale(diff_gen_no_w_pipe, diffs_fft_np.min(), diffs_fft_np.max())
row['diff-scaled-to-diff-ffts   BASE=gen_w   vary=pipe'] = scale(diff_gen_w_pipe, diffs_fft_np.min(), diffs_fft_np.max())
row['diff-scaled-to-diff-ffts   BASE=gen_w   vary=prompt'] = scale(diff_gen_w_prompt, diffs_fft_np.min(), diffs_fft_np.max())

# scale again, only to itself
row['diff-scaled-to-itself   BASE=gen_w   PIPE=orig   vary=initial_noise'] = scale(diff_gen_w_orig_initial_noise, diff_gen_no_w_pipe.min(), diff_gen_no_w_pipe.max())
row['diff-scaled-to-itself   BASE=gen_w   PIPE=alt   vary=initial_noise'] = scale(diff_gen_w_alt_initial_noise, diff_gen_no_w_pipe.min(), diff_gen_no_w_pipe.max())
row['diff-scaled-to-itself   BASE=gen_no_w   vary=pipe'] = scale(diff_gen_no_w_pipe, diff_gen_no_w_pipe.min(), diff_gen_no_w_pipe.max())
row['diff-scaled-to-itself   BASE=gen_w   vary=pipe'] = scale(diff_gen_w_pipe, diff_gen_w_pipe.min(), diff_gen_w_pipe.max())
row['diff-scaled-to-itself   BASE=gen_w   vary=prompt'] = scale(diff_gen_w_prompt, diff_gen_w_prompt.min(), diff_gen_w_prompt.max())

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