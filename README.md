# Histology

IF USING SLIDECHAT:
```
conda create --name slidechat python=3.10 -y
conda activate slidechat
cd slidechat
pip install -e .
cd CONCH
pip install -e .
pip install deepspeed
conda config --set channel_priority flexible
conda env update -n slidechat -f environment.yml
```
