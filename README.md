
<p align="center">
  <img src="https://github.com/user-attachments/assets/f95d8fec-9793-4d91-a8d0-7caaa607de8d" />
</p>

# Scratch Transformer
Reimplementation of the original 2017 "Attention is all you need" transformer from 2017. For a detailed overview, see [my long blog post](https://www.ubaada.com/post/fc9c5fc3).

The encoder-deocder base model with apprx. 63M was trained on WMT-14 English-to-German dataset. I tried to match the original paper where possible.

## Training
To train, simply run:
```
python train.py
```
It will:
1. Download the WMT14-de-en dataset from kagglehub.
2. Train a BPE tokenizer and save it to 'wmt14_de_en_tokenizer.json'
3. Load the downloaded dataset.
4. Load the base with base config.
5. Load any checkpoint from previous epoch if found in 'model_weights'
6. Start training the model while logging to wandb.


The defaults model config is hardcoded for base model. The default training params are optimised for RTX 4070 12GB. Change the params according to your GPU before running. On this setup, it achieves approximately 60-80% GPU utilisation. 

The training script is using a simple PyTorch dataloader. It can be improved upon by implementing a bucket loader which batches only similar length inputs. It is not currently implemented.
