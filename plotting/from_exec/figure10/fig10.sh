#!/bin/bash

name=fig10_$(date -Iseconds)

LOG_FILES=`pwd`/$name
mkdir -p $LOG_FILES

# puzzle
cd $AEROOT/puzzle
./ae/figure10/fig10.sh $LOG_FILES
cd -

mkdir -p logs/
cp -r $LOG_FILES/* ./logs
