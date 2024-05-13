#!/bin/bash

python3 -m pip install torch==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
python3 -m pip install tensorboard==2.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple