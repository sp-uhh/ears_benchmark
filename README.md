# EARS-WHAM and EARS-Reverb generation scripts

This repository contains generation scripts for the EARS-WHAM and EARS-Reverb benchmarks.

Please make sure you have installed the required packages.

You can run the following command to install them:

```
python -m pip install -r requirements.txt
```

## Generate EARS-WHAM

To generate EARS-WHAM, first download the original EARS and WHAM! datasets with the following command, where `<data_dir>` is the directory where the datasets will be downloaded:

```
bash download_ears_wham.sh <data_dir>
```

Then you can generate the EARS-WHAM dataset with the following command:

```
python generate_ears_wham.py  --data_dir <data_dir>
```

## Generate EARS-Reverb

Comming soon...