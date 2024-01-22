## PM2.5

XGBoost performance in different settings:

| Model Settings | Train_RMSE | Train_Pearson_R  | Test_RMSE | Test_Pearson_R |
| ------------- |:-------------:| :-----:| :-----:| :-----:|
| Original + lat_long | 37.4940 | 0.8986 | 55.4154 | 0.8399 |
| Original + normalized + lat_long | 36.8489 | 0.8965 | 56.7666 | 0.8411 |
| Original  + timestamp | 19.8799 | 0.8336 | 131.2294 | 0.5288 |
| Original + normalized + timestamp | 19.8799 |  0.8336 | 131.2294 | 0.5288 |
| New Params + lat_long | 10.9958 | 0.9672 | 14.5034 | 0.9512 |
| New Params + normalized + lat_long | 10.9699 |  0.9688 | 15.0432 | 0.9440 |
| New Params + timestamp | 9.9678 | 0.9285 | 47.1918 | 0.9380 |
| New Params + normalized + timestamp | 9.9646 |  0.9297 | 44.8590 | 0.9392 |