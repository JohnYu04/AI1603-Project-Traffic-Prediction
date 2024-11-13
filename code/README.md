# GAT+Transformer Traffic Prediction Model

This repository contains the implementation of the GAT+Transformer model designed for traffic prediction using the Paris traffic probe dataset. The model integrates Graph Attention Networks (GAT) with the Transformer architecture to effectively leverage spatial and temporal data for enhanced traffic forecasting accuracy.

## Dataset

The dataset comprises traffic flow data collected from 208 traffic probes across Paris from January 1, 2016, to November 30, 2016. Traffic flow is measured in vehicles per hour. Additionally, adjacency information `paris_adj` is provided, with weights representing the normalized shortest path lengths between probes.

### Data Split
- **Training Set (`train.csv`)**: Data from January 1, 2016, to July 1, 2016.
- **Validation Set (`val.csv`)**: Data from July 1, 2016, to August 1, 2016.
- **Test Set (`test_history.csv`)**: Data from August 1, 2016, to November 30, 2016. Data points are omitted every 24 hours for prediction.
- **Prediction Format (`format.csv`)**: Specifies the format for the final prediction data, with values to be filled in between commas.

## Model

The GAT+Transformer model combines the spatial feature extraction capabilities of GATs with the temporal processing power of Transformers. This approach allows for capturing complex dependencies in traffic data, enhancing prediction performance significantly over traditional methods.

### Architecture
- **Graph Attention Networks (GAT)**: Captures spatial relationships between traffic probes.
- **Transformer**: Utilizes self-attention mechanisms to process temporal dependencies.

## Requirements

- Python 3.x
- PyTorch
- NumPy

## Usage

To train and evaluate the model, run:
```
python train.py
```
