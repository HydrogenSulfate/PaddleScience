# How to train/inference

## 1. train FAE

``` sh
python main.py
```

then we get the pretrained FAE weights in `outputs_gifm_fae/xxx/xxx/latest.pdparams`

## 2. train DiT

``` sh
python main.py -cn gifm_diffusion.yaml FAE.pretrained_model_path=outputs_gifm_fae/xxx/xxx/latest.pdparams
```

then we get the pretrained FAE weights in `outputs_gifm_dit/xxx/xxx/latest.pdparams`

## 3. inference

after train FAE and DiT, we can:
1. use **FAE.encoder** to encode the given physical field,
2. integrate from 0 to 1 with **DiT**,
3. use **FAE.decoder** to decode the integrated result to prediction field.

``` sh
python main.py -cn gifm_diffusion.yaml mode=eval EVAL.pretrained_model_path=outputs_gifm_dit/xxx/xxx/checkpoints/latest
```

preview result(not fully trained):
![result](<outputs_gifm_dit/2025-07-14/09-25-49/Pressure predition of sample 1~4.png>)
