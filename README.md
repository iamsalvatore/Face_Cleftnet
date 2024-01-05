
# Cleft Palate Classification Project

## Overview
This project aims to classify cleft palate disease in individuals using a deep learning approach. It leverages an Attention ResNet architecture for enhanced feature extraction and classification accuracy. The models are stored and managed efficiently using HDF5 format for optimized performance.

## Repository Structure

```
cleft-palate-classification/
│
├── model/
│   ├── attention_resnet_model.h5
│   └── [other model files]
│
├── hdf5_helper/
│   ├── hdf5_converter.py
│   └── [additional helper scripts]
│
├── data/
│   ├── raw_images/
│   └── hdf5_data/
│       ├── train.h5
│       ├── test.h5
│       └── validate.h5
│
├── scripts/
│   └── [additional scripts]
│
├── README.md
└── requirements.txt
```

## Setup and Installation

1. **Clone the Repository**: `git clone [repository-url]`
2. **Install Dependencies**: `pip install -r requirements.txt`

## Usage

- **Converting Images to HDF5 Format**:
  - Run `python hdf5_helper/hdf5_converter.py` to convert raw images into HDF5 format.
  - The converted files will be stored in `data/hdf5_data/`.

- **Model Training and Evaluation**:
  - Load the models from the `model/` directory.
  - Train the Attention ResNet model using the HDF5 dataset.
  - Evaluate the model performance on the test dataset.

## Contributing

Contributions to this project are welcome. Please follow the standard pull request process for contributions.

## License

[Specify the License Here]
