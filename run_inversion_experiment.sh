python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id_alt "stabilityai/stable-diffusion-2-1-base"
python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id_alt "stabilityai/stable-diffusion-2-1-base" --w_injection 'seed'    --w_pattern "seed_ring"

python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id_alt "stabilityai/stable-diffusion-2-1-base" --w_injection 'complex' --w_pattern "zeros"       --w_radius 25
python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id_alt "stabilityai/stable-diffusion-2-1-base" --w_injection 'seed'    --w_pattern "seed_zeros"  --w_radius 25

python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY"  --model_id_alt "stabilityai/stable-diffusion-2-1-base" --w_injection 'complex' --w_pattern "zeros"       --w_radius 25
python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY"  --model_id_alt "stabilityai/stable-diffusion-2-1-base" --w_injection 'seed'    --w_pattern "seed_zeros"  --w_radius 25


#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY" --model_id_alt "stabilityai/stable-diffusion-2-1-base"
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY" --model_id_alt "hakurei/waifu-diffusion"
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY" --model_id_alt "runwayml/stable-diffusion-v1-5"
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY" --model_id_alt "prompthero/openjourney"
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY" --model_id_alt "Fictiverse/Stable_Diffusion_Microscopic_model"



#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY" --model_id_alt "stabilityai/stable-diffusion-2-1-base"  --w_injection 'complex' --w_pattern "zeros"       --w_radius 25
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY" --model_id_alt "stabilityai/stable-diffusion-2-1-base"  --w_injection 'seed'    --w_pattern "seed_zeros"  --w_radius 25
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY" --model_id_alt "stabilityai/stable-diffusion-2-1-base"  --w_injection 'seed'    --w_pattern "seed_zeros"  --w_radius 32  --w_mask_shape  "square"
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id "KANDINSKY" --model_id_alt "stabilityai/stable-diffusion-2-1-base"  --w_injection 'complex' --w_pattern "zeros"       --w_radius 32  --w_mask_shape "square"



#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id_alt "stabilityai/stable-diffusion-2-1-base"
#
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id_alt "hakurei/waifu-diffusion"
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id_alt "runwayml/stable-diffusion-v1-5"
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id_alt "prompthero/openjourney"
#python run_inversion_experiment.py --prompt_id_in_dataset 2 --model_id_alt "Fictiverse/Stable_Diffusion_Microscopic_model"
#
#python run_inversion_experiment.py --swap_pipes --prompt_id_in_dataset 2 --model_id_alt "hakurei/waifu-diffusion"
#python run_inversion_experiment.py --swap_pipes --prompt_id_in_dataset 2 --model_id_alt "runwayml/stable-diffusion-v1-5"
#python run_inversion_experiment.py --swap_pipes --prompt_id_in_dataset 2 --model_id_alt "prompthero/openjourney"
#python run_inversion_experiment.py --swap_pipes --prompt_id_in_dataset 2 --model_id_alt "Fictiverse/Stable_Diffusion_Microscopic_model"





#python run_inversion_experiment.py --gen_seed 0 --prompt_id_in_dataset 0
#python run_inversion_experiment.py --gen_seed 0 --prompt_id_in_dataset 1
#python run_inversion_experiment.py --gen_seed 1 --prompt_id_in_dataset 0
#python run_inversion_experiment.py --gen_seed 1 --prompt_id_in_dataset 1