# Histology
Fine-tuning a large vision-language model (SlideChat) to analyze and predict 15 different pathology features in IBD slides to assist pathologists.
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
