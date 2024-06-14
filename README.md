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
python generate_ears_wham.py --data_dir <data_dir>
```

## Generate EARS-Reverb

To generate EARS-Reverb, first download the original EARS and RIR datasets with the following command, where `<data_dir>` is the directory where the datasets will be downloaded:

```
bash download_ears_reverb.sh <data_dir>
```

Then you can generate the EARS-Reverb dataset with the following command:

```
python generate_ears_reverb.py --data_dir <data_dir>

```

# License

The code and dataset are released under [CC-NC 4.0 International license](https://github.com/facebookresearch/ears_dataset/blob/main/LICENSE).

# References

If you use the dataset or any derivative of it, please cite our [research paper](https://arxiv.org/abs/2406.06185):

```
@inproceedings{richter2024ears,
  title={{EARS}: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation},
  author={Richter, Julius and Wu, Yi-Chiao and Krenn, Steven and Welker, Simon and Lay, Bunlong and Watanabe, Shinjii and Richard, Alexander and Gerkmann, Timo},
  booktitle={ISCA Interspeech},
  year={2024}
}
```

For audio samples, visit the [project page](https://sp-uhh.github.io/ears_dataset/).