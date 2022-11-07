# EEG_NJU_dataset
Containing EEG relative codes and download method for dataset

## Usage
1. Download the dataset from [https://zenodo.org/record/7253438]. Extract them to any directory. Please make sure the directory ONLY CONTAINS english characters. Please DO NOT use Chinese or other special characters in the path.
2. Download codes from github. Extract them to any directory.
3. Process the dataset using `sliceTo2D.m` or `sliceTo2D_KUL.m`, which will split the dataset into 5 folds. Everything is automatic, please wait for several hours. You can specify how many window length you want the script to process.
4. Start your training and evaluation by changing the `srcdir` in the beginning of the `.py` file and type `python convmap_attention.py` or `python simple_cnn.py` in your terminal
