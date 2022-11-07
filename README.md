# EEG_NJU_dataset
Containing EEG relative codes and download method for dataset. The relative models are implemented by PyTorch 1.10.

## Usage
1. Download the dataset from [https://zenodo.org/record/7253438]. Extract them to any directory. Please make sure the directory ONLY CONTAINS english characters. Please DO NOT use Chinese or other special characters in the path.
2. Download codes from github. Extract them to any directory. Please make sure the directory ONLY CONTAINS english characters. Please DO NOT use Chinese or other special characters in the path.
3. Process the dataset using `sliceTo2D_NJU.m` or `sliceTo2D_KUL.m` in the `process_script` folder, which will split the dataset into 5 folds. Everything is automatic, please wait for several hours. You can specify how many window length you want the script to process. Other settings can be found inside the script.
4. Start your training and evaluation by changing the `srcdir` in the beginning of the `.py` file and type `python LSM.py` or `python CNN_baseline.py` in your terminal.

## Explanation
The `model` folder contains some utilities and necessary sub-models and is required to run the python file.
