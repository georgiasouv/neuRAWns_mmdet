# ========================================
kinit -r 7d souval_g  # give password
krenew -a -K 30 -b

klist # verify that both are running
ps aux | grep krenew # check background process

sbatch cluster_scripts/exp003.sh