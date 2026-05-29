#!/bin/bash
# =============================================================================
# neuRAWns_mmdet — Environment Setup Script
# =============================================================================
# This script is split into two clearly labelled sections:
#
#   SECTION A: Run on your LOCAL machine to export the conda environment
#   SECTION B: Run on any NEW machine (cluster, lab machine, etc.) to reproduce
#               the environment and get experiments running
#
# Usage:
#   - Do NOT run this entire script top to bottom
#   - Read the section headers and run only the commands relevant to your machine
#
# Requirements on any machine:
#   - Anaconda or Miniconda installed
#   - Git configured with SSH access to your GitHub
#   - CUDA drivers installed (version may vary — see note in SECTION B)
# =============================================================================


# =============================================================================
# SECTION A — LOCAL MACHINE ONLY
# Run these commands ONCE on your local machine to export your working environment
# =============================================================================

# Step A1: Activate your working conda environment
# conda activate mmdet12

# Step A2: Export the environment to a file inside your repo
# The --no-builds flag strips OS-specific build strings so the file works
# across different Linux machines (local, cluster, lab machine, etc.)
# conda env export --no-builds > ~/Desktop/neuRAWns_mmdet/environment.yml

# Step A3: Commit the environment file to your repo so it travels with your code
# cd ~/Desktop/neuRAWns_mmdet
# git add environment.yml
# git commit -m "Add conda environment file for reproducibility"
# git push origin main

# NOTE: You only need to re-run Section A when you add or remove packages
# from your conda environment. Keep environment.yml up to date.


# =============================================================================
# SECTION B — NEW MACHINE SETUP
# Run these commands on any new machine (cluster, lab machine, etc.)
# to reproduce the exact environment and get experiments running
# =============================================================================

# Step B1: Clone your repo
# git clone git@github.com:georgiasouv/neuRAWns_mmdet.git
# cd neuRAWns_mmdet

# Step B2: Check the CUDA version available on this machine
# Your PyTorch build must match the CUDA version of the machine you're on
# nvidia-smi

# Step B3 (IMPORTANT): If the CUDA version on this machine DIFFERS from your
# local machine, you must edit environment.yml before creating the environment.
# Find the torch line in environment.yml and note the CUDA suffix (e.g. cu121).
# grep "torch" environment.yml
#
# If the CUDA versions don't match, install PyTorch manually AFTER step B4:
# See: https://pytorch.org/get-started/locally/
# Example for CUDA 11.8:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Example for CUDA 12.1:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step B4: Recreate the conda environment from the exported file
# This installs all packages at the exact same versions as your local machine
# conda env create -f environment.yml

# Step B5: Activate the environment
# conda activate mmdet12

# Step B6: Install mmdetection as an editable (local) install
# This step is always required on every new machine — conda env create does NOT
# handle editable installs automatically because they point to a local path
# pip install -e mmdetection/

# Step B7: Verify mmdet resolves to the correct path inside your repo
# You should see: .../neuRAWns_mmdet/mmdetection/mmdet/__init__.py
# python -c "import mmdet; print(mmdet.__file__)"

# Step B8: You're ready. Run your experiments as normal.
# =============================================================================