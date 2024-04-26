# Dataset

- `locations.txt`: index | block | district | longitude | latitude
- `bihar_may_jan.npy`: Numpy array of shape `[t, l, f]` where `t` are the number of timestamps, `l` are the total number of stations and `f` are the number of features of each station.
    - `Feature List`: Relative Humidity, Temperature, Boundary Layer Height, U component of wind (U10), V component of wind (V10), K-Index, Surface Pressure, Total Precipitation, PM2.5