#!/bin/bash
# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=NSTPP
#SBATCH --output train_gpu.out.%j                        
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=5GB
#SBATCH --time=6:00:00
##SBATCH --dependency=afterany:37086792

module load py-pytorch/1.6.0_py36

sed -i "s/^my_device =.*/my_device = 'gpu'/" train_stpp.py

# srun python3 train_stpp.py --data earthquakes_jp --spatial_dim=2 --model attncnf --tpp neural --l2_attn --num_iterations 10000 --max_events 1000 --experiment_id JP_spatial_resume2 --data_file ./data/earthquakes/eq_JP_m4.0_spatial.npz --resume ./experiments/attncnf64-64-64_JP_spatial_resume1/model.pth

# srun python3 train_stpp.py --data eq_mag --spatial_dim=3 --model attncnf --tpp neural --l2_attn --num_iterations 10000 --max_events 1000 --experiment_id JP_mag4_resume --data_file ./data/earthquakes/eq_JP_m4.0.npz --resume ./experiments/attncnf64-64-64_JP_mag4_resume1/model.pth

srun python3 train_stpp.py --data eq_mag --spatial_dim=2 --model attncnf --tpp neural --l2_attn --num_iterations 10000 --max_events 1000 --experiment_id JP_mag4_space2_resume3 --data_file ./data/earthquakes/eq_JP_m4.0.npz --resume ./experiments/attncnf64-64-64_JP_mag4_space2_resume2/model.pth