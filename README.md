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
