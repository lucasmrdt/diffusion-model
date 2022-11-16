# Benchmarks

## Batch Size

| batch_size |       time        |
| :--------: | :---------------: |
|    2048    |   15.42 s/epoch   |
|    1024    |   16.26 s/epoch   |
|  **512**   | **18.10 s/epoch** |
|    256     |   23.34 s/epoch   |
|    128     |   33.82 s/epoch   |
|     64     |   55.66 s/epoch   |
|     32     |  100.33 s/epoch   |

We should use a batch size of **512** because it's a good trade-off between GPU RAM and time.

# Cmd

```bash
screen -S "mu-tild-l2" -dm ./train.py --loss=mu-tild-l2 --epochs=1000 --batch_size=512;
screen -S "x-prev-sig" -dm ./train.py --loss=x-prev-l2 --epochs=100 --batch_size=512 --scheduler=linear-gamma-bar;

screen -S "x-prev-linear-noise" -dm ./train.py --loss=x-prev-l2 --epochs=100 --batch_size=512 --scheduler=linear-noise;
screen -S "x-prev-linear-x" -dm ./train.py --loss=x-prev-l2 --epochs=100 --batch_size=512 --scheduler=linear-x;
screen -S "x-prev-cosine" -dm ./train.py --loss=x-prev-l2 --epochs=100 --batch_size=512 --scheduler=cosine;

screen -S "x-prev-01" -dm ./train.py --loss=x-prev-l2 --epochs=100 --batch_size=512 --normalize_range=0,1;
screen -S "x-prev-0255" -dm ./train.py --loss=x-prev-l2 --epochs=100 --batch_size=512 --normalize_range=0,255;
screen -S "optuna" -dm ./train.py --epochs=50 --optuna;

./generate.py --model_id=3767016b1bd404c84d05b4fe083d2d6c94171747 --grid;
./generate.py --model_id=38139a585fce461f46bf8d852da9f61688133422 --grid;
```

# Score

|                 model_id                 |        FID         |  Precision   |   Recall    |
| :--------------------------------------: | :----------------: | :----------: | :---------: |
| 09593b8aa5cc97196cbe3d9f33ca8da9a60d2423 | 24.447352257962507 | 0.3779296875 | 0.19140625  |
| 0990623cddd911a710bbc398e040718fe6dfb584 | 43.25397445062495  | 0.2431640625 | 0.107421875 |

09593b8aa5cc97196cbe3d9f33ca8da9a60d2423 -> FID: 18.912463302436265 (1024)
-> FID: 14.041892065052991 (60000)
-> FID: 0.030252694025587945 (60000+dims=64)

0990623cddd911a710bbc398e040718fe6dfb584 -> FID: 23.392960888701424 (1024)
-> FID: 21.14389089247078 (2048)
-> FID: 19.78453120816181 (60000)
-> FID: 0.008670651386033679 (60000+dims=64)
VGG

- differente archi (dense, conv, u-net)
- time steps
- noising process
- nb_channels
- optionnal label embedding
