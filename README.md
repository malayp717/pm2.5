# Spatio-Temporal Forecasting of $$PM_{2.5}$$ via Spatial-Diffusion Guided Encoder-Decoder Architecture

This repository contains the code for the research paper **"Spatio-Temporal Forecasting of $$PM_{2.5}$$ via Spatial-Diffusion guided Encoder-Decoder Architecture"** accepted for presentation at **"CODS-COMAD'24"**.

## Table of Contents
- [Abstract](Abstract)
- [Dataset](Dataset)
- [Requirements](Requirements)
- [Run](Run)
- [Reference](Reference)

## Abstract
In many problem settings that require spatio-temporal forecasting, the values in the time-series not only exhibit spatio-temporal correlations but are also influenced by spatial diffusion across locations. One such example is forecasting the concentration of fine particulate matter ($$PM_{2.5}$$) in the atmosphere which is influenced by many complex factors, the most important ones being diffusion due to meteorological factors as well as transport across vast distances over a period of time. We present a novel Spatio-Temporal Graph Neural Network architecture, that specifically captures these dependencies to forecast the $$PM_{2.5}$$ concentration. Our model is based on an encoder-decoder architecture where the encoder and decoder parts leverage gated recurrent units (GRU) augmented with a graph neural network (TransformerConv) to account for spatial diffusion. Our model can also be seen as a generalization of various existing models for time-series or spatio-temporal forecasting. We demonstrate the model's effectiveness on two real-world $$PM_{2.5}$$ datasets: (1) data collected by us using a recently deployed network of low-cost $$PM_{2.5}$$ sensors from 511 locations spanning the entirety of the Indian state of Bihar over a period of one year, and (2) another publicly available dataset that covers severely polluted regions from China for a period of 4 years. Our experimental results show our model's impressive ability to account for both spatial as well as temporal dependencies precisely. The code is publicly available at [Github Repository](https://github.com/malayp717/pm2.5).

## Dataset
This study is based on two similar real world datasets:
- **Bihar, India:** A dataset collected using 511 low-cost sensors deployed across the entire region of Bihar for a year (01/05/2023 - 30/04/2024).
- **KnowAir Dataset:** Dataset covering severly polluted regions in China, spanning across 4 years (01/01/2015 - 31/12/2018). Dataset is available at [KnowAir](https://github.com/shuowang-ai/PM2.5-GNN/tree/main).

## Requirements
### Prerequisites
- `Python 3.10.13` or above
- GPU recommended for training (trained on NVIDIA A30 with 24G memory)

### Installation
- Clone this repository
    ```
    git clone https://github.com/malayp717/pm2.5.git
    cd pm2.5
    ```
- Install dependencies:
    ```
    pip install -r requirements.txt
    ```

### Experimental Setup
- Choose the model you want to run, and make required changes in either `bihar_config.yaml` or `china_config.yaml`, depending on your location preference.
    ```
    train:
        model: GRU
        model: GC_GRU
        model: GraphConv_GRU
        model: GNN_GRU
        model: Attn_GNN_GRU
    ```

## Run
### Training
    python -u train.py --config <config_file>.yaml
    
Make sure to choose the correct configuration file:
- `bihar_config.yaml`: Use this file for training on the Bihar dataset.
- `china_config.yaml`: Use this file for training on the China dataset.

### Evaluation
    python -u stats.py --config <config_file>.yaml

## Reference