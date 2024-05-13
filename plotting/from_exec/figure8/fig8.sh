#!/bin/bash

name=fig8_$(date -Iseconds)

LOG_FILES=`pwd`/$name
mkdir -p $LOG_FILES

# -------- dschat --------
cd $AEROOT/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/
./fig8.sh $LOG_FILES
cd -
# ------------------------


# -------- puzzle --------
cd $AEROOT/puzzle
# > setup ray cluster here, $NODELIST pass from `./RUNME-b.sh`
NNODES=$(echo $NODELIST | tr ',' '\n' | wc -l)
srun -u -N $NNODES -w $NODELIST --gres=gpu:8 -p Nvidia_A800 --mem=500G --cpus-per-task=32 bash setup_ray_cluster.sh &
sleep 10;
SLURM_RAY_SETUP_INFO=$(squeue -u `whoami` -t RUNNING -o "%.18i %.9P %.200j")
export SLURM_RAY_SETUP_JOB_ID=$(echo "$SLURM_RAY_SETUP_INFO" | awk 'NR==2{print $1}')
echo "setup ray done ..., SLURM_RAY_SETUP_JOB_ID=${SLURM_RAY_SETUP_JOB_ID}"

./ae/figure8/fig8.sh $LOG_FILES
cd -
# ------------------------

mkdir -p logs/
cp -r $LOG_FILES/* ./logs
