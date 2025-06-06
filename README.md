# Spatio-Temporal Forecasting of $$PM_{2.5}$$ via Spatial-Diffusion Guided Encoder-Decoder Architecture

This repository contains the code for the research paper [**"Spatio-Temporal Forecasting of PM2.5 via Spatial-Diffusion guided Encoder-Decoder Architecture"**](https://arxiv.org/abs/2412.13935) accepted for presentation at [CODS-COMAD'24].

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
- Choose the model you want to run, and make required changes in either `configs/bihar.yaml` or `configs/china.yaml`, depending on your location preference.
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
- `bihar.yaml`: Use this file for training on the Bihar dataset.
- `china.yaml`: Use this file for training on the China dataset.

### Evaluation
    python -u stats.py --config <config_file>.yaml

## Citation
```
@misc{pandey2024spatiotemporalforecastingpm25spatialdiffusion,
      title={Spatio-Temporal Forecasting of PM2.5 via Spatial-Diffusion guided Encoder-Decoder Architecture}, 
      author={Malay Pandey and Vaishali Jain and Nimit Godhani and Sachchida Nand Tripathi and Piyush Rai},
      year={2024},
      eprint={2412.13935},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.13935}, 
}
```