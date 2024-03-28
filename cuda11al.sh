#!/bin/bash

#SBATCH --mail-user=shunyu.wu@students.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --job-name="al"
#SBATCH --account=ws_00000
##SBATCH --partition=gpu-invest
##SBATCH --gres=gpu:rtx3090:1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --chdir=/storage/homefs/sw21c033/AL/simple_AL
#SBATCH --mem-per-cpu=12G
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
##SBATCH --output=/storage/homefs/sw21c033/AL/simple_AL/results/test.txt
#SBATCH --error=/storage/homefs/sw21c033/AL/simple_AL/error.txt

# Put your code below this line
module load Anaconda3
eval "$(conda shell.bash hook)"
#conda activate torch_cuda11
#conda activate torch
module load CUDA/11.3.1
#conda activate sims
conda activate cuda11
#conda activate actsegmul

find ./ -path ./results -prune -o -name "*.py" -exec cp --parents {} results/script/ \;
#python main_multiClassAL_stage2.py
#python main_multiClassAL.py
python main.py
