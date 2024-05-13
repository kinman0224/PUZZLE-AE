#!/bin/bash

CLUSTER="orion"

# clean the log/ inside the plotting/from_exec/
./clean.sh

start_time=$(date +%s)

echo "The script starts now: $(date -Iseconds)"

export AEROOT=`pwd`
output_dir=$AEROOT/outputs_from_exec_$(date -Iseconds)
mkdir $output_dir

# The nodelist used in puzzle's experiment ray cluster setup.
# !!! Modify the nodelist if necessary.
export NODELIST=gpu20,gpu21,gpu9,gpu10

# ---- Figure 8 ----
cd ./plotting/from_exec/figure8
./fig8.sh
cd -
python3 ./plotting/from_exec/plot_fig8.py ./plotting/from_exec/figure8/logs $output_dir

# ---- Figure 9 ----
cd ./plotting/from_exec/figure9
./fig9.sh
cd -
python3 ./plotting/from_exec/plot_fig9.py ./plotting/from_exec/figure9/logs $output_dir

# ---- Figure 10 ----
cd ./plotting/from_exec/figure10
./fig10.sh
cd -
python3 ./plotting/from_exec/plot_fig10.py ./plotting/from_exec/figure10/logs $output_dir

# ---- Figure 11 ----
cd ./plotting/from_exec/figure11
./fig11.sh
cd -
python3 ./plotting/from_exec/plot_fig11.py ./plotting/from_exec/figure11/logs $output_dir

# ---- Figure 12 ----
cd ./plotting/from_exec/figure12
./fig12.sh
cd -
python3 ./plotting/from_exec/plot_fig12.py ./plotting/from_exec/figure12/logs $output_dir

# ---- Table 3 ----
cd ./plotting/from_exec/table3
./table3.sh
cd -
python3 ./plotting/from_exec/plot_table3.py ./plotting/from_exec/table3/logs $output_dir

# ----

end_time=$(date +%s)

echo "The script ends now: $(date -Iseconds)"

# ---- close ray cluster ----
# make sure all slurm jobs have been canceled
SLURM_RAY_SETUP_INFO=$(squeue -u `whoami` -t RUNNING -o "%.18i %.9P %.200j")
export SLURM_RAY_SETUP_JOB_ID=$(echo "$SLURM_RAY_SETUP_INFO" | awk 'NR==2{print $1}')
echo "!!! [Not graceful... but work...] Force delete ray ..., SLURM_RAY_SETUP_JOB_ID=${SLURM_RAY_SETUP_JOB_ID}"
if [ "$SLURM_RAY_SETUP_JOB_ID" != "" ]; then
    scancel -Q $SLURM_RAY_SETUP_JOB_ID 2> /dev/null
fi
# ----

duration=$((end_time - start_time))
echo "Total duration: ${duration} seconds"
