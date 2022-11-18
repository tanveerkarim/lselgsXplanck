#!/bin/bash
#SBATCH --qos debug
#SBATCH -C cpu
#SBATCH -A desi
#SBATCH -J dndz-uncertainty
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=64 
#SBATCH --mem=40G ##request full node memory
#SBATCH --array=9
#SBATCH -e /pscratch/sd/t/tanveerk/temp/log/dndz-uncertainty_20221118_%A_%a.err ###write out errors
#SBATCH -o /pscratch/sd/t/tanveerk/temp/log/dndz-uncertainty_20221118_%A_%a.out ###write out print #logs
#SBATCH --mail-user=tanveer.karim@cfa.harvard.edu
#SBATCH --mail-type=ALL

ID=$SLURM_ARRAY_TASK_ID
source ${HOME}/.bashrc ##check if this or bashrc
module load python
conda activate myenv_perlmuter

script_home='/global/homes/t/tanveerk/lselgsXplanck/src/'
echo $ID
python $script_home/photoz-bootstrap.py --JID $ID