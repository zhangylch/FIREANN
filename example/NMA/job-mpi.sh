#!/bin/sh
#PBS -V
#PBS -q a100
#PBS -N nma-2
#PBS -l nodes=1:ppn=8
#export CUDA_VISIBLE_DEVICES="4"
source /public/home/group_zyl/.bashrc
# conda environment
conda_env=pt200
export OMP_NUM_THREADS=8
#path to save the code
path="/public/home/group_zyl/zyl/program/EFREANN-EMA/reann/"

#Number of processes per node to launch
NPROC_PER_NODE=1

#Number of process in all modes
WORLD_SIZE=`expr $PBS_NUM_NODES \* $NPROC_PER_NODE`

MASTER=`/bin/hostname -s`

#MPORT=`ss -tan | awk '{print $5}' | cut -d':' -f2 | \
#        grep "[2-9][0-9]\{3,3\}" | sort | uniq | shuf -n 1`

#You will want to replace this
COMMAND="$path "
conda activate $conda_env 
cd $PBS_O_WORKDIR 
python3 -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --max_restarts=0 --nnodes=$PBS_NUM_NODES --rdzv_id=$PBS_JOBID --rdzv_backend=c10d --rdzv_endpoint=$MASTER:34113 $COMMAND > out
#python3 -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --max_restarts=0 --nnodes=$PBS_NUM_NODES --standalone $COMMAND > out

