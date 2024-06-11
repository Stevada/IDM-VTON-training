"""Download weights from HF repo."""

# pip install --upgrade huggingface_hub
git config --global credential.helper store

huggingface-cli login

apt-get update 
apt-get install git-lfs 
git lfs install 

mkdir checkpoints
cd checkpoints
git clone https://huggingface.co/yisol/IDM-VTON
