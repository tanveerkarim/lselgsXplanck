#!/bin/bash
#SBATCH --qos shared
#SBATCH -C cpu
#SBATCH -A desi
#SBATCH -J photoz_rf_hyperparam
#SBATCH -t 03:35:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=60G ##request full node memory
#SBATCH -e /pscratch/sd/t/tanveerk/temp/log/photoz_hyper_%A_%a.err ###write out errors
#SBATCH -o /pscratch/sd/t/tanveerk/temp/log/photoz_hyper_%A_%a.out ###write out print #logs
#SBATCH --mail-user=tanveer.karim@cfa.harvard.edu
#SBATCH --mail-type=ALL

source ${HOME}/.bashrc ##check if this or bashrc
module load python
conda activate myenv_perlmutter

script_home='/global/homes/t/tanveerk/lselgsXplanck/analysis/'
python $script_home/photoz_rf_hyperparam_cv.py