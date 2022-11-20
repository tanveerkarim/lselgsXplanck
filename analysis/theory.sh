#!/bin/bash
#SBATCH --qos regular
#SBATCH -C cpu
#SBATCH -A desi
#SBATCH -J theory
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=256
#SBATCH --mem=105G ##request full node memory
#SBATCH -e /pscratch/sd/t/tanveerk/temp/log/Cl+Dl_20221119_WMAX2048_%A_%a.err ###write out errors
#SBATCH -o /pscratch/sd/t/tanveerk/temp/log/Cl+Dl_20221119_WMAX2048_%A_%a.out ###write out print #logs
#SBATCH --mail-user=tanveer.karim@cfa.harvard.edu
#SBATCH --mail-type=ALL

source ${HOME}/.bashrc ##check if this or bashrc
module load python
conda activate myenv_perlmutter

script_home='/global/homes/t/tanveerk/lselgsXplanck/analysis'
echo $ID
python $script_home/generateTheory.py