# PUZZLE AE

## Abstract

This artifact helps to reproduce the results of paper #124 at USENIX ATC'24, titled PUZZLE: Efficiently Aligning Large Language Models through Light-Weight Context Switch.

The repository includes:

- Log files in `logs/` and scripts in `plotting/`. These were used for plotting the figures and some Out-of-the-box scripts are provided for running experiments.
- Source code in `puzzle/` and `DeepSpeedExamples/`. Both puzzle and baselines codes are provided.

It should be noticed that reproducing the original PUZZLE experiments requires strict hardware requirements: 32 NVIDIA A100 PCIe and NVLink GPUs. To overcome hardware limitations, we have made available the raw data of PUZZLE experiments and scripts for reproducing experiments on 32 NVIDIA A100 PCIe GPUs.



## Scope

The artifact can be used to reproduce the experiments of the paper, including the end-to-end speedup (Figure 8), intra-stage ablation study (Figure 9 and 10), inter-stage ablation study (Figure11), and Fine-Grained Performance Breakdown (Figure 12 and Table 3) .



## Getting Started

### Quick Reproduction: Plotting from Logs (~2minutes)

**Hardware requirments: No GPUs are required.**

**Software requirements: Python and some commonly used packages (e.g. Matplotlib and NumPy).**

The Only command needed to plot all figures in the evaluation section is:

```bash
source env.sh
./RUNME-a.sh
```

This may takes a few minutes to complete. Once you have successfully run this command, you will find a directory named `outputs_from_log_${TIME_YOU_RUN}`  which contains the generated figures `figure[8-12].pdf` and `table3.pdf`.

In detail, the `RUNMME-a.sh` will read log files, perform some post-processing, and plot the figures.


### In-depth Reproduction: Plotting from Real Run (~2hours)

**Hardware requirements: 32 NVIDIA A100 PCIe GPUs.**

**Software requirements: Distributed training software (e.g. CUDA, NCCL and PyTorch) and plotting software (e.g. Matplotlib and NumPy).**

Once your environment is ready, simply run the following Out-of-the-box command to reproduce the experiments:

```bash
source env.sh
./RUNME-b.sh
```

This may takes several hours to complete. Once you have successfully run this command, you will find a directory named `outputs_from_exec_${TIME_YOU_RUN}`  which contains the generated figures and table.

The final output logs would like:
```bash
The script ends now: 20xx-0x-0xT20:53:07+0800
!!! [Not graceful... but is ok] Force delete ray ..., SLURM_RAY_SETUP_JOB_ID=xxxxx
srun: Force Terminated job xxxxx
Total duration: xxxxx seconds
[test@admin1 ae]$ srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
slurmstepd: error: *** STEP 111988.0 ON gpu20 CANCELLED AT 20xx-xx-xxTxx:xx:xx ***
...
```
> Please ignore the error message, as we forced the Slurm job to cancel. Just press **Enter** again, and it will be okay.

**Note**: Please note that the provided scripts have only been tested on a cluster managed by the [Slurm](https://www.schedmd.com/) scheduler and [Spack](https://github.com/spack/spack/) package manager. If your cluster uses different management software, modifications to the scripts may be necessary.



## Installation

**For AE Reviewers**: Please follow the instructions in "Comments for AEC" on HotCRP and skip this section if you want to use the provided environment. We strongly recommend using the provided environment because some network issues may occur in the provided cluster (e.g., connecting to GitHub). All model weights (or config.) have been prepared on the provided cluster.

### Clone this repository and navigate to PUZZLE-AE folder

```bash
git clone https://github.com/kinman0224/PUZZLE-AE.git
cd PUZZLE-AE
```

### PUZZLE

It requires CUDA, NCCL, and PyTorch (v1.10.0). To install PyTorch, please use the following command:

```  bash
pip install torch==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### Megatron-LM

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) was used as the GPT training framework in PUZZLE's experiments, and NVIDIA Apex is required. However, the apex package on PyPI is broken, so you need to install it from source. Clone it from https://github.com/NVIDIA/apex, and install it using the following command:

```bash
python3 setup.py install --user --cuda_ext --cpp_ext
```

### DeepSpeed

[DeepSpeed](https://github.com/kinman0224/DeepSpeed.git) `v0.10.4` was used in DeepSpeed's experiments. We have fixed some bug within the Hybrid Engine and we provided the code host at https://github.com/kinman0224/DeepSpeed, banch `fix_tp`, you can use the following command:

```bash
git clone https://github.com/kinman0224/DeepSpeed.git -b fix_tp
```
This command will clone the repository from the specified branch (fix_tp) where the bug fixes are implemented.

### DeepSpeed-Chat

To install the requirements for DeepSpeed-Chat, you can use the following command:
```bash
cd PUZZLE-ae
cd DeepSpeedExamples/applications/DeepSpeed-Chat
pip install -r ./requirements.txt
```

### Model Weights

Please check out huggingface Model Zoo for all public [LLaMA](https://huggingface.co/meta-llama) checkpoints.



## Discussion

Due to hardware limitations, we can only provide the A800 PCIe to reproduce the `orion` part and portions of the result. The artifact can be used to reproduce the experiments described in the paper, including the end-to-end comparison (Figure 8), intra-stage ablation study (Figures 9 and 10), inter-stage ablation study (Figure 11), and performance breakdown (Figures 12 and Table 3).

### Section 7.2: End-to-End Speedup

In the paper, sec.7.2 has figure 8 as its results. However, in the reproduction, due to hardware limitations, **only the first cluster (labeled as `on orion`) in the figure 8 are plotted**.

### Section 7.3: Intra-stage Ablation Study

In the paper, sec.7.3 has figure 9 and 10 as its results. However, in the reproduction, **only the first cluster (labeled as `on orion`) in the figure 9 and left part of Figure10 (7B/7B with batch size = 128 and 13B/7B with batch size = 64) are plotted**.

Due to hardware and time limitations, we have decided not to reproduce the other data in figure 9 and 10. First, the complete experiment takes several hours and is very time-consuming. Second, performing reproduction on a cluster with NVIDIA A100 GPUs can be expensive.

### Section 7.4: Inter-stage Ablation Study

In the paper, sec.7.4 has fiugre 11 as its results. However, in the reproduction, due to hardware limitations,  **only the first cluster (labeled as `on orion`) in the figure 11 are plotted**. Nonetheless, the selected portion is representative enough to explain the effectiveness of PUZZLE.

### Section 7.5: Fine-Grained Performance Breakdown

In the paper, sec.7.5 has figure 12 and table 3 as its results. However, in the reproduction, due to hardware limitations,  **only the first cluster (labeled as `on orion`) in the figure 12 and table 3 are plotted**.
