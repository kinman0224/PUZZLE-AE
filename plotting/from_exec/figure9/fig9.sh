#!/bin/bash

name=fig9_$(date -Iseconds)

LOG_FILES=`pwd`/$name
mkdir -p $LOG_FILES

# puzzle
cd $AEROOT/puzzle
./ae/figure9/fig9.sh $LOG_FILES
cd -

mkdir -p logs/
cp -r $LOG_FILES/* ./logs
