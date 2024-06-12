#!/bin/bash

# Check if the correct number of arguments was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <data_dir>"
    exit 1
fi

data_dir=$1
data_dir="$(realpath -s "$data_dir")"
echo "Data directory is set to $data_dir"

# Download the EARS speech dataset
if [ -d "$data_dir/EARS" ]; then
    echo "[Warning] Skip download of EARS. The directory $data_dir/EARS arleady exists."
else
    mkdir $data_dir/EARS
    for X in $(seq -w 001 107); do
        wget -L https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p${X}.zip -O $data_dir/EARS/p${X}.zip
        n_files=`unzip -l $data_dir/EARS/p${X}.zip | tail -n 1 | xargs echo -n | cut -d' ' -f2`
        unzip $data_dir/EARS/p${X}.zip -d $data_dir/EARS | tqdm --unit files --unit_scale --total $n_files > /dev/null
        rm $data_dir/EARS/p${X}.zip
    done
fi

# Download WHAM!48kHz noise dataset
if [ -d "$data_dir/WHAM48kHz" ]; then
    echo "[Warning] Skip download of WHAM48kHz. The directory $data_dir/WHAM48kHz arleady exists."
else
    if [ -f "$data_dir/WHAM48kHz.zip" ]; then
        echo "[Warning] $data_dir/WHAM48kHz.zip already exists. Skip download."
    else
        echo "Download WHAM48kHz noise dataset to $data_dir/WHAM48kHz.zip"
        wget -O $data_dir/WHAM48kHz.zip https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/high_res_wham.zip
    fi
    mkdir $data_dir/WHAM48kHz;
    echo "Extract $data_dir/WHAM48kHz.zip to $data_dir/WHAM48kHz"
    n_files=`unzip -l $data_dir/WHAM48kHz.zip | tail -n 1 | xargs echo -n | cut -d' ' -f2`
    unzip $data_dir/WHAM48kHz.zip -d $data_dir/WHAM48kHz | tqdm --unit files --unit_scale --total $n_files > /dev/null
    rm $data_dir/WHAM48kHz.zip
fi
