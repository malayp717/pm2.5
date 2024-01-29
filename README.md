## PM2.5

XGBoost performance in different settings:

| Model Settings | Train_RMSE | Train_Pearson_R  | Test_RMSE | Test_Pearson_R |
| ------------- |:-------------:| :-----: | :-----: | :-----:|
| Original + lat_long | 36.4563 | 0.9054 | 57.618 | 0.8419 |
| Original + normalized + lat_long | 35.4955 | 0.9018 | 58.5243 | 0.8426 |
| Original  + timestamp | 14.0318 | 0.7944 | 148.2336 | 0.0139 |
| Original + normalized + timestamp | 14.1292 | 0.7941 | 144.453 | 0.0596 |
| New Params + lat_long | 10.7503 | 0.9701 | 15.9811 | 0.943 |
| New Params + normalized + lat_long | 10.6443 | 0.9693 | 15.7042 | 0.9404 |
| New Params + timestamp | 8.6201 | 0.8978 | 77.6517 | 0.7746 |
| New Params + normalized + timestamp | 8.4899 | 0.8993 | 78.9822 | 0.7498 |