if [[ $(hostname) == *"wmg"* ]]; then
    ln -sf ~/Desktop/mmdetection/tools tools
else
    ln -sf ~/mmdetection/tools tools  # cluster path
fi