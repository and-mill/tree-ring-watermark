import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

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

import matplotlib.pyplot as plt


MODELS = [
    'kandinsky-community/kandinsky-2-1',
    'stabilityai/stable-diffusion-2-1-base',
    'runwayml/stable-diffusion-v1-5',
    'prompthero/openjourney',
    'Fictiverse/Stable_Diffusion_Microscopic_model',
    'hakurei/waifu-diffusion',
    #'dalle-mini/dalle-mega',
]

UNET_ATTRS = [
    'conv_in.bias',
    'conv_in.weight',
    'conv_norm_out.bias',
    'conv_norm_out.weight',
    'conv_out.bias',
    'conv_out.weight',
    'time_embedding.linear_1.bias',
    'time_embedding.linear_1.weight',
    'time_embedding.linear_2.bias',
    'time_embedding.linear_2.weight',
    ]


def get_nested_attr(obj, attr_path):
    """
    Get a nested attribute from an object using a dot-separated string.

    Parameters:
    obj (object): The object to get the attribute from.
    attr_path (str): Dot-separated string specifying the nested attribute.

    Returns:
    object: The value of the nested attribute.
    """
    attrs = attr_path.split('.')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def get_pipe(model: str):

    if 'KANDINSKY' in model.upper():
        pipe = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1",
                                                 torch_dtype=torch.float32,
                                                 variant='fp16')
    else:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            model,
            scheduler=DPMSolverMultistepScheduler.from_pretrained(model, subfolder='scheduler'),
            torch_dtype=torch.float32,
            variant='fp16',  # updated after new transformers version
            safety_checker=None, requires_safety_checker=False,
            )
    return pipe
