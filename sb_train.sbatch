#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=neural-editor 
#SBATCH --output=sl_neural_editor.out
#SBATCH --error=sl_neural_editor.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=6
#SBATCH --mem=140GB
#SBATCH --gres=gpu:2
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

#module purge # unload all modules
#module load cuda/8.0.44
#module load cudnn/8.0v5.1

SEED=1000
source venv_editor/bin/activate

export TEXTMORPH_DATA=/misc/vlgscratch4/BowmanGroup/pmh330/neural-editor/neural-editor-data
PYTHONPATH=. python2 textmorph/edit_model/main.py configs/edit_model/edit_baseline.txt $SEED >> out_train_savae$SEED.log 2>&1

