#!/bin/bash

rm -rf logs/*
mkdir -p logs/

# copy 7b/350m logs from figure8
cp ../figure8/logs/llama_7b_350m_* ./logs

# copy 13b/350m logs from figure8
cp ../figure8/logs/llama_13b_350m_* ./logs

# copy 33b/7b logs from figure8
cp ../figure8/logs/llama_33b_7b_* ./logs
