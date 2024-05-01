# BIOSCAN-6M

Here is the code that supports experiments with BIOSCAN-CLIP.

## Set environment
For now, you can set the environment by typing
```shell
conda create -n bioscan-clip-6M python=3.10
conda activate bioscan-clip-6M
conda install pytorch=2.0.1 torchvision=0.15.2 torchtext=0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
conda install -c conda-forge faiss

```
in the terminal. However, based on your GPU version, you may have to modify the torch version and install other packages manually in difference version.
## Download data
```shell
# From project folder
mkdir -p data/BioScan_1M/split_data
cd data/BioScan_1M/split_data
wget https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/data/version_0.2.1/BioScan_data_in_splits.hdf5
unzip clip_data.zip
```

## Download checkpoint for BarcodeBERT
```shell
# From project folder
mkdir -p ckpt/BarcodeBERT/5_mer
cd ckpt/BarcodeBERT/5_mer
wget https://aspis.cmpt.sfu.ca/projects/bioscan/clip_project/ckpt/BarcodeBERT/model_41.pth
```

## Train
```shell
# From project folder
python scripts/train_cl.py 'model_config={config_name}'
```
For example
```shell
# From project folder
python scripts/train_cl.py 'model_config=lora_vit_lora_barcode_bert_ssl.yaml'
```
For multiple GPU, you may have to
```shell
NCCL_P2P_LEVEL=NVL python scripts/train_cl.py 'model_config=lora_vit_lora_barcode_bert_ssl.yaml'
```

## Eval
```shell
# From project folder
python scripts/inference_and_eval.py 'model_config=lora_vit_lora_barcode_bert_lora_bert_ssl'
```