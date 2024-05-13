#!/bin/bash

name=fig11_$(date -Iseconds)

LOG_FILES=`pwd`/$name
mkdir -p $LOG_FILES

# puzzle
cd $AEROOT/puzzle
./ae/figure11/fig11.sh $LOG_FILES "orion"
cd -

mkdir -p logs/

# the exec logs
cp $LOG_FILES/* ./logs

# copy 7b/350m logs from figure8
cp ../figure8/logs/llama_7b_350m_opt* ./logs
find ../figure8/logs -name "llama_7b_350m_base*" -exec sh -c 'cp {} ./logs/$(basename {} | sed "s/base/train_opt/")' \;

# copy 7b/7b logs from figure8
cp ../figure8/logs/llama_7b_7b_opt* ./logs
find ../figure8/logs -name "llama_7b_7b_base*" -exec sh -c 'cp {} ./logs/$(basename {} | sed "s/base/train_opt/")' \;

# copy 33b/7b logs from figure8
cp ../figure8/logs/llama_33b_7b_opt* ./logs
find ../figure8/logs -name "llama_33b_7b_base*" -exec sh -c 'cp {} ./logs/$(basename {} | sed "s/base/train_opt/")' \;
