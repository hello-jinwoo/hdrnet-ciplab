# Application of HDRnet to Color Constancy
- HDRnet from repository https://github.com/creotiv/hdrnet-pytorch

Python 3.6

### Dependencies

To install the Python dependencies, run:

    pip install -r requirements.txt
    
## Datasets
    LSMI_refined dataset

## Usage
    
To train a model, run the following command:

    python train.py --dataset {PATH_TO_DATASET} --epochs {NUM_EPOCHS} --batch-size {BATCH_SIZE} --ckpt-path {PATH_TO_CHECKPOINT_TO_BE_SAVED}
    
    
To test image run:

    python test.py --dataset {PATH_TO_DATASET} --ckpt {PATH_TO_CHECKPOINT}
    
