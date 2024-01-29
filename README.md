## PM2.5

XGBoost performance in different settings:

| Model Settings | Train_RMSE | Train_Pearson_R  | Train_RMSE | Train_Pearson_R | Test_RMSE | Test_Pearson_R |
| ------------- |:-------------:| :-----: | :-----: | :-----: | :-----: | :-----:|
| Original + lat_long | 36.4563 | 0.9054 | 59.4008 | 0.8309 | 57.618 | 0.8419 |
| Original + normalized + lat_long | 35.4955 | 0.9018 | 62.9432 | 0.8043 | 58.5243 | 0.8426 |
| Original  + timestamp | 14.0318 | 0.7944 | 39.2807 | 0.1884 | 148.2336 | 0.0139 |
| Original + normalized + timestamp | 14.1292 | 0.7941 | 40.3806 | 0.1638 | 144.453 | 0.0596 |
| New Params + lat_long | 10.7503 | 0.9701 | 15.5754 | 0.9408 | 15.9811 | 0.943 |
| New Params + normalized + lat_long | 10.6443 | 0.9693 | 16.5046 | 0.9264 | 15.7042 | 0.9404 |
| New Params + timestamp | 8.6201 | 0.8978 | 14.8442 | 0.8367 | 77.6517 | 0.7746 |
| New Params + normalized + timestamp | 8.4899 | 0.8993 | 14.6879 | 0.8412 | 78.9822 | 0.7498 |