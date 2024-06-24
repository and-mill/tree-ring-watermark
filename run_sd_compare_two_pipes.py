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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


sd_1 = InversableStableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    scheduler=DPMSolverMultistepScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='scheduler'),
    torch_dtype=torch.float32,
    variant='fp16',  # updated after new transformers version
    safety_checker=None, requires_safety_checker=False,
    )
sd_1 = sd_1.to(device)


sd_2 = InversableStableDiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2-1-base',
    scheduler=DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler'),
    torch_dtype=torch.float32,
    variant='fp16',  # updated after new transformers version
    safety_checker=None, requires_safety_checker=False,
    )
sd_2 = sd_2.to(device)


print()