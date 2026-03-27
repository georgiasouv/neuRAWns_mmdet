#!/bin/bash
# =============================================================================
# pre_submit_checklist.sh
# Run these commands IN ORDER before every sbatch submission
# See pre_submit_checklist.txt for explanation of each command
# =============================================================================

# ── STEP 1 — Kerberos ticket ──────────────────────────────────────────────────
klist

# If klist shows no ticket or expired, run this:
# kinit -r 7d souval_g < ~/.kerberos_pass

# ── STEP 2 — Data mount ───────────────────────────────────────────────────────
ls /cifs/Shares/WMGData/

# ── STEP 3 — How long will my job run? ───────────────────────────────────────
# Answer this yourself before step 4:
#   < 1h  (quick test / sanity check)  → use test or medium
#   > 4h  (real training run)          → use xlong

# ── STEP 4 — Check partition availability ────────────────────────────────────
sinfo -s

# ── STEP 5 — Check free GPUs ─────────────────────────────────────────────────
sinfo -o "%20N %10P %10t %20G" | grep -E "idle|mix"

# ── STEP 6 — Submit ──────────────────────────────────────────────────────────
# Replace exp012.sh with your actual script
sbatch ~/neuRAWns_mmdet/cluster_scripts/exp012.sh

# ── STEP 7 — Confirm it is queued ────────────────────────────────────────────
squeue -u $USER --format="%.10i %.20j %.8T %.10M %.10l %R"


# =========================================================================================================================== #
# =========================================================================================================================== #
# =========================================================================================================================== #


cluster_help() {
cat << 'EOF'
╔══════════════════════════════════════════════════════════════════╗
║              CLUSTER UTILS — QUICK REFERENCE                    ║
╠══════════════════════════════════════════════════════════════════╣
║  INTERACTIVE SESSIONS                                            ║
║    idev              — interactive GPU session (1h, medium)      ║
║    idev_long         — interactive GPU session (4h, xlong)       ║
║    idev_cpu          — interactive CPU-only session              ║
║                                                                  ║
║  CLUSTER STATUS                                                  ║
║    gpu_free          — show free GPUs per node                   ║
║    partitions        — show all partitions and time limits       ║
║    node_info         — show all nodes + status                   ║
║    who_is_using      — show all running jobs by user             ║
║                                                                  ║
║  MY JOBS                                                         ║
║    myjobs            — show all my running/pending jobs          ║
║    mylog <jobid>     — tail the output log for a job             ║
║    myerr <jobid>     — tail the error log for a job              ║
║    cancel <jobid>    — cancel a job or array                     ║
║    cancel_all        — cancel ALL my jobs (asks confirmation)    ║
║                                                                  ║
║  QUICK CHECKS                                                    ║
║    check_gpu         — test CUDA is visible (run inside idev)    ║
║    check_data        — test CIFS dataset mount is alive          ║
║    check_env         — print conda env, python, torch, mmdet     ║
║    disk_usage        — show my quota on networkhome              ║
║                                                                  ║
║  KERBEROS                                                        ║
║    kauth             — authenticate (reads ~/.kerberos_pass)     ║
║    kcheck            — show current ticket status + expiry       ║
╚══════════════════════════════════════════════════════════════════╝
EOF
}

# ===============================================================================================

sinfo -s                                    # Shows partition names, max time limits, and how many nodes are available/busy. 
sinfo -o "%20N %10P %10t %20G" | grep gpu   # Shows every GPU node, which partition it's in, its state (idle/mix/alloc), and what GPU type it has. 
                                            # This is what you look at before writing --gres=gpu:1 in your script.
squeue -p medium --format="%.10i %.15u %.8T %.10M" | grep -c RUNNING    # How loaded is a specific partition? Counts how many jobs are actively running on medium
sinfo -o "%20N %30G" | grep gpu             # Some clusters have mixed GPU types (V100, A100, etc.). 
                                            # This tells you what's actually in each node so you can request the right one with --gres=gpu:v100:1 if needed.








#!/bin/bash
# =============================================================================
# cluster_utils.sh — SLURM aliases for neuRAWns HPC
# Usage: source ~/cluster_utils.sh
# To auto-load: echo "source ~/cluster_utils.sh" >> ~/.bashrc
# =============================================================================
 
# ── MY JOBS ──────────────────────────────────────────────────────────────────
alias myjobs='squeue -u $USER --format="%.10i %.20j %.8T %.10M %.10l %R"'
alias killjobs='scancel -u $USER'
 
# ── CLUSTER STATUS ───────────────────────────────────────────────────────────
alias partitions='sinfo --format="%-15P %-10a %-10l %-6D %-8t" | sort'
alias freegpus='sinfo -o "%20N %10P %10t %20G" | grep -E "idle|mix"'
alias whosusing='squeue --format="%-12u %-20j %-10T %-12M %R" --sort="-M" | head -30'
 
# ── STORAGE & ENVIRONMENT ────────────────────────────────────────────────────
alias checkmount='ls /cifs/Shares/WMGData/'
alias myticket='klist'
alias myenvs='conda env list'
alias diskusage='du -sh ~ ~/neuRAWns_mmdet /cifs/Shares/WMGData/ 2>/dev/null'
 
# ── INTERACTIVE SESSIONS ─────────────────────────────────────────────────────
# 1 GPU, 4h — default debug session
alias idev='srun --partition=medium --gres=gpu:1 --time=4:00:00 --pty bash'
# CPU only — for inspecting data, testing imports
alias idev_cpu='srun --partition=medium --time=2:00:00 --pty bash'
 
echo "cluster_utils.sh loaded — aliases active"
 