import os

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

import torch

from optim_utils import *
from io_utils import *

from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

from andmill_utils import get_pipe, get_nested_attr, MODELS, UNET_ATTRS


device = 'cuda' if torch.cuda.is_available() else 'cpu'


DIR = 'sd_unet_similarity_matrices'


def kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """
    Compute the KL divergence between two Gaussian distributions.
    """
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
    term3 = -0.5
    return term1 + term2 + term3

def compute_similarity_matrix(vectors):
    # Number of vectors
    n = len(vectors)
    
    # Create a matrix for storing similarity values
    similarity_matrix = np.zeros((n, n))
    
    # Compute means and variances
    means = {key: vec.mean().item() for key, vec in vectors.items()}
    variances = {key: vec.var().item() for key, vec in vectors.items()}
    
    # Compute pairwise KL divergence and fill the matrix
    keys = list(vectors.keys())
    for i in range(n):
        for j in range(i, n):
            mu_p, sigma_p = means[keys[i]], np.sqrt(variances[keys[i]])
            mu_q, sigma_q = means[keys[j]], np.sqrt(variances[keys[j]])
            
            # Compute KL divergence
            kl_div = kl_divergence_gaussian(mu_p, sigma_p, mu_q, sigma_q)
            
            # Store the similarity; for KL divergence, lower values mean more similarity
            similarity = 1 / (1 + kl_div)  # Convert divergence to similarity
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
    
    return similarity_matrix, keys


# DO
pipes = [get_pipe(model) for model in MODELS]
for attribute in UNET_ATTRS:

    # get specific weights for all models
    weights_maps = {model: get_nested_attr(pipe.unet, attribute).flatten().detach().cpu() for model, pipe in zip(MODELS, pipes)}
    weights_maps_norm = {k: v / v.abs().sum() for k, v in weights_maps.items()}
    
    # Compute the similarity matrix
    #similarity_matrix, labels = compute_similarity_matrix(weights_maps)
    similarity_matrix, labels = compute_similarity_matrix(weights_maps_norm)
    

    # PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT PLOT    
    # Plotting the similarity matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title(attribute)
    plt.tight_layout()
    
    # Show the plot
    plt.savefig(os.path.join(DIR, f'{attribute}.png'))
    plt.clf()
