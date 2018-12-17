#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=neural-editor 
#SBATCH --output=sl_neural_editor.out
#SBATCH --error=sl_neural_editor.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=pmh330@nyu.edu
#SBATCH --mail-type=END,FAIL

SEED=$1
numiter=$2
eps=$3
momentum=$4

export TEXTMORPH_DATA=/scratch/nn1119/neural-editor-data
module load python2/intel/2.7.12
source ~/pytorch_neural/py2.7.12/bin/activate

if [ $# -ge 2 ]; then
   PYTHONPATH=. python2 textmorph/edit_model/main.py configs/edit_model/edit_baseline.txt $SEED >> out_train_savae$SEED.log 2>&1
else
   PYTHONPATH=. python2 textmorph/edit_model/main.py configs/edit_model/edit_baseline.txt $SEED --num-iter $numiter --eps $eps --momentum $momentum >> out_train_savae$SEED.log 2>&1
fi