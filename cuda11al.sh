#!/bin/bash
#SBATCH --mail-user=shunyu.wu@students.unibe.ch
#SBATCH --mail-type=start,end,fail
#SBATCH --job-name="cudaal"
#SBATCH --account=ws_00000
##SBATCH --partition=gpu-invest
##SBATCH --gres=gpu:rtx3090:1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --chdir=/storage/homefs/sw21c033/AL/simple_AL
#SBATCH --mem-per-cpu=12G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/storage/homefs/sw21c033/AL/simple_AL/results/test.txt
#SBATCH --error=/storage/homefs/sw21c033/AL/simple_AL/error.txt

# Put your code below this line
module load Anaconda3
eval "$(conda shell.bash hook)"
module load CUDA/11.3.1
conda activate cuda11

find ./ -path ./results -prune -o -name "*.py" -exec cp --parents {} results/script/ \;
python main.py
