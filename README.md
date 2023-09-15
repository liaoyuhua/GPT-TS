# GPT-TS
Reproduction of the results of the paper "One Fits All: Power General Time Series Analysis by Pretrained LM".


<div align=center> <image src="./assets/struct.png" width="400px"> </div>


**Performance**

| dataset | multivariate | seq_len | pred_len |  lr  | layers | patch_size/stride |  mae   |
| :-----: | :---------: | :-----: | :------: | :--: | :----------------: | :---------------------: | :----: |
|  ETTH1  |    True     |   384   |    96    | 1e-3 |         6          |          16/8           | 0.4819 |
|  ETTH1  |    False    |   384   |    96    | 1e-3 |         6          |          16/8           | 0.2641 |


**Dataset**

ETTh1, ETTh2, ETTm1, ETTm2 are put in the folder "data". PyTorch dataset classes have been implemented, you can check them in `src/dataset.py`.

**Model weights**

they would be put in the folder "gpt2" automatically or you can download files from the link: https://huggingface.co/gpt2/tree/main, and put them in folder "gpt2".



**Compared to last version**, this version has the following changes:

> support multivariate time series

> drop temporal embedding to align with original paper

> add patch embedding

> add RevIN layer to ease drift

> option for batch normalization in transformer layer

> change training and inference approach to generate the whole sequence at once (not autoregressive)

> option for FlattenHead or PoolingHead for sequential predictions

> support auto mixed precision for speed up