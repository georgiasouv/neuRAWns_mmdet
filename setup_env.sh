if [[ $(hostname) == *"wmg"* ]]; then
    ln -sf ~/Desktop/mmdetection/tools tools
else
    ln -sf  /networkhome/WMGDS/souval_g/mmdetection/tools tools  # cluster path
fi