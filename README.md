# Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust

<img src=scripts/teaser.png  width="80%" height="60%">

This code is the official implementation of [Tree-Ring Watermarks](http://arxiv.org/abs/2305.20030).

If you have any questions, feel free to email Yuxin (<ywen@umd.edu>).

## About
We propose Tree-Ring Watermarking to watermark diffusion model outputs. Tree-Ring Watermarking chooses the initial noise array so that its Fourier transform contains a carefully constructed pattern near its center. This pattern is called the key. This initial noise vector is then converted into an image using the standard diffusion pipeline with no modifications. To detect the watermark in an image, the diffusion model is inverted to retrieve the original noise array used for generation. This array is then checked to see whether the key is present.

## Dependencies
- PyTorch == 1.13.0
- transformers == 4.23.1
- diffusers == 0.11.1
- datasets

Note: higher diffusers version may not be compatible with the DDIM inversion code.

and-mill: 1st environment (treering) try:
- conda install pytorch torchvision -c pytorch
    - because of [W NNPACK.cpp:61] Could not initialize NNPACK! Reason: Unsupported hardware. -> USE_NNPACK=0 python setup.py install (https://discuss.pytorch.org/t/bug-w-nnpack-cpp-80-could-not-initialize-nnpack-reason-unsupported-hardware/107518)

Maybe necessary for running jupyters correctly
- conda installc nb_conda_kernels
- python -m ipykernel install --user --name treering --display-name "Python (notebook_env)"
    - Assuming env is named treering

Try this at the end, because GPU may not be visible:
- conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

Did this after trying alternative models for inversion
- pip install --upgrade diffusers transformers
- pip uninstall diffusers
- pip install diffusers==0.21.1

and-mill: 2nd environment (treering_1) try:
run
- python setup.py build
- python setup.py install
- pip install -r requirements.txt

- newer numpies may cause thought wandb: AttributeError: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead.
- pip install numpy==1.24
Then do:
- conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

- pip install IPyhton

## Usage

### Perform main experiments and calculate CLIP Score
For non-adversarial case, you can simply run:
```
python run_tree_ring_watermark.py --run_name no_attack --w_channel 3 --w_pattern ring --start 0 --end 1000 --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k
```

You can modify arguments to perform attack. For example, for rotation with 75 degrees:
```
python run_tree_ring_watermark.py --run_name rotation --w_channel 3 --w_pattern ring --r_degree 75 --start 0 --end 1000 --with_tracking
```

For more adversarial cases, see [here](scripts/tree_ring.sh).

For other watermark types mentioned in the paper, you can check [scripts/](scripts/).

### Calculate FID
You can download 5000 COCO examples used in the paper [here](https://drive.google.com/drive/folders/1saWx-B3vJxzspJ-LaXSEn5Qjm8NIs3r0?usp=sharing). Feel free to add more data or other datasets according to the format of `fid_outputs/coco/meta_data.json`.

Then, to calculate FID, you may run:
```
python run_tree_ring_watermark_fid.py --run_name fid_run --w_channel 3 --w_pattern ring --start 0 --end 5000 --with_tracking --run_no_w
```

### Perform main experiments for Imagenet Models
You can get the pre-trained models [here](https://github.com/openai/guided-diffusion). For example, the link of the model used by the paper is [256x256 diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt). Then, you may follow the [script](scripts/tree_ring_imagenet.sh) to run the experiments.

## Parameters
Crucial hyperparameters for Tree-Ring:

- `w_channel`: the index of the watermarked channel. If set as -1, watermark all channels.
- `w_pattern`: watermark type: zeros, rand, ring.
- `w_radius`: watermark radius.

## Suggestions and Pull Requests are welcome!
