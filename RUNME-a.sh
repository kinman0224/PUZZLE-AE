#!/bin/bash

start_time=$(date +%s)

echo "The script starts now: $(date -Iseconds)"

export AEROOT=`pwd`
output_dir=$AEROOT/outputs_from_logs_$(date -Iseconds)
mkdir $output_dir

LOG_FILE=./logs

python3 ./plotting/from_logs/plot_fig8.py $LOG_FILE/figure8/ $output_dir
python3 ./plotting/from_logs/plot_fig9.py $LOG_FILE/figure9/ $output_dir
python3 ./plotting/from_logs/plot_fig10.py $LOG_FILE/figure10/ $output_dir
python3 ./plotting/from_logs/plot_fig11.py $LOG_FILE/figure11/ $output_dir
python3 ./plotting/from_logs/plot_fig12.py $LOG_FILE/figure12/ $output_dir

python3 ./plotting/from_logs/plot_table3.py $LOG_FILE/table3/ $output_dir

end_time=$(date +%s)

echo "The script ends now: $(date -Iseconds)"

duration=$((end_time - start_time))
echo "Total duration: ${duration} seconds"
