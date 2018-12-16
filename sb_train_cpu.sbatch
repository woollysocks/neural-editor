#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=cpu-editor
#SBATCH --mail-type=END
##SBATCH --mail-user=pmh330@nyu.edu


#module purge # unload all modules
#module load cuda/8.0.44
#module load cudnn/8.0v5.1

SEED=1111
source venv_editor/bin/activate

export TEXTMORPH_DATA=/misc/vlgscratch4/BowmanGroup/pmh330/neural-editor/neural-editor-data
PYTHONPATH=. python2 textmorph/edit_model/main.py configs/edit_model/edit_baseline.txt >> out_train_cpu.log 2>&1

