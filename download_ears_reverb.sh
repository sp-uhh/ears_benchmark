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

# Download ACE-Challenge Dataset
if [ -d "$data_dir/ACE-Challenge" ]; then
    echo "[Warning] Skip download of ACE-Challenge. The directory $data_dir/ACE-Challenge arleady exists."
else
    mkdir $data_dir/ACE-Challenge;
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_Data.tbz2
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_instructions_v01.pdf
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_Microphone_arrangements_v02.pdf
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_RIRN_Chromebook.tbz2
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_RIRN_Crucif.tbz2
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_RIRN_EM32.tbz2
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_RIRN_Lin8Ch.tbz2
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_RIRN_Mobile.tbz2
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_RIRN_Single.tbz2
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_Software.tbz2
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_Corpus_Speech.tbz2
    wget -P $data_dir/ACE-Challenge/ https://zenodo.org/records/6257551/files/ACE_TASLP_ref.bib
    for file in "$data_dir/ACE-Challenge/"*.tbz2; do
        echo "Extract $file"
        extract_dir="${file%%.tbz2}"
        mkdir -p "$extract_dir"
        tar -jxvf "$file" -C "$extract_dir" > /dev/null
        rm "$file"
    done
fi

# Download AIR dataset
if [ -d "$data_dir/AIR" ]; then
    echo "[Warning] Skip download of AIR. The directory $data_dir/AIR arleady exists."
else
    if [ -f "$data_dir/AIR.zip" ]; then
        echo "[Warning] $data_dir/AIR.zip already exists. Skip download."
    else
        echo "Download AIR noise dataset to $data_dir/AIR.zip"
        wget -O $data_dir/AIR.zip https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/air_database_release_1_4.zip
    fi
    mkdir $data_dir/AIR;
    echo "Extract $data_dir/AIR.zip to $data_dir/AIR"
    n_files=`unzip -l $data_dir/AIR.zip | tail -n 1 | xargs echo -n | cut -d' ' -f2`
    unzip $data_dir/AIR.zip -d $data_dir/AIR | tqdm --unit files --unit_scale --total $n_files > /dev/null
    rm $data_dir/AIR.zip 
    mkdir $data_dir/AIR/AIR_1_4/AIR_wav_files
    echo "Extract $data_dir/AIR/AIR_1_4/AIR_wav_files.zip to $data_dir/AIR/AIR_1_4/AIR_wav_files"
    n_files=`unzip -l $data_dir/AIR/AIR_1_4/AIR_wav_files.zip | tail -n 1 | xargs echo -n | cut -d' ' -f2`
    unzip $data_dir/AIR/AIR_1_4/AIR_wav_files.zip -d $data_dir/AIR/AIR_1_4/AIR_wav_files | tqdm --unit files --unit_scale --total $n_files > /dev/null
    rm $data_dir/AIR/AIR_1_4/AIR_wav_files.zip 
fi

# Download DetmoldSRIR dataset
if [ -d "$data_dir/DetmoldSRIR" ]; then
    echo "[Warning] Skip download of DetmoldSRIR. The directory $data_dir/DetmoldSRIR arleady exists."
else
    if [ -f "$data_dir/DetmoldSRIR.zip" ]; then
        echo "[Warning] $data_dir/DetmoldSRIR.zip already exists. Skip download."
    else
        echo "Download DetmoldSRIR noise dataset to $data_dir/DetmoldSRIR.zip"
        wget -O $data_dir/DetmoldSRIR.zip https://zenodo.org/api/records/4116247/files-archive
    fi
    mkdir $data_dir/DetmoldSRIR;
    echo "Extract $data_dir/DetmoldSRIR.zip to $data_dir/DetmoldSRIR"
    n_files=`unzip -l $data_dir/DetmoldSRIR.zip | tail -n 1 | xargs echo -n | cut -d' ' -f2`
    unzip $data_dir/DetmoldSRIR.zip -d $data_dir/DetmoldSRIR | tqdm --unit files --unit_scale --total $n_files > /dev/null
    rm $data_dir/DetmoldSRIR.zip 
    echo "Extract $data_dir/DetmoldSRIR/DetmoldSRIR_v01.zip to $data_dir/DetmoldSRIR/"
    n_files=`unzip -l $data_dir/DetmoldSRIR/DetmoldSRIR_v01.zip | tail -n 1 | xargs echo -n | cut -d' ' -f2`
    unzip $data_dir/DetmoldSRIR/DetmoldSRIR_v01.zip -d $data_dir/DetmoldSRIR/ | tqdm --unit files --unit_scale --total $n_files > /dev/null
    rm $data_dir/DetmoldSRIR/DetmoldSRIR_v01.zip
fi

# Download dEchorate dataset
if [ -d "$data_dir/dEchorate" ]; then
    echo "[Warning] Skip download of dEchorate. The directory $data_dir/dEchorate arleady exists."
else
    gdown --folder https://drive.google.com/drive/folders/1yGTh_BjnVNwDgBsn5mkuW3i4rJIgZwlS -O $data_dir/
    for file in `ls $data_dir/dEchorate/sofa/*.zip`; do
        n_files=`unzip -l $file | tail -n 1 | xargs echo -n | cut -d' ' -f2`
        unzip $file -d $data_dir/dEchorate/sofa/ | tqdm --unit files --unit_scale --total $n_files > /dev/null
        rm $file
    done
fi

# Download BRUDEX dataset
if [ -d "$data_dir/BRUDEX" ]; then
    echo "[Warning] Skip download of BRUDEX. The directory $data_dir/BRUDEX arleady exists."
else
    if [ -f "$data_dir/BRUDEX.zip" ]; then
        echo "[Warning] $data_dir/BRUDEX.zip already exists. Skip download."
    else
        echo "Download BRUDEX noise dataset to $data_dir/BRUDEX.zip"
        wget -O $data_dir/BRUDEX.zip https://zenodo.org/records/8340195/files/rir.zip?download=1
    fi
    mkdir $data_dir/BRUDEX;
    echo "Extract $data_dir/BRUDEX.zip to $data_dir/BRUDEX"
    n_files=`unzip -l $data_dir/BRUDEX.zip | tail -n 1 | xargs echo -n | cut -d' ' -f2`
    unzip $data_dir/BRUDEX.zip -d $data_dir/BRUDEX | tqdm --unit files --unit_scale --total $n_files > /dev/null
    rm $data_dir/BRUDEX.zip 
fi

# Download Palimpsest dataset
if [ -d "$data_dir/Palimpsest" ]; then
    echo "[Warning] Skip download of Palimpsest. The directory $data_dir/Palimpsest arleady exists."
else
    gdown 1utDu8wCdpj6fj0AlXNMXMIeWgn93arEF -O $data_dir/Palimpsest.zip
    unzip -o $data_dir/Palimpsest.zip -d $data_dir > /dev/null
    mv $data_dir/'Sonic Palimpsest -Impulse Response Library' $data_dir/Palimpsest
    rm $data_dir/Palimpsest.zip
    rm -r $data_dir/__MACOSX
fi

# Download ARNI dataset
if [ -d "$data_dir/ARNI" ]; then
    echo "[Warning] Skip download of ARNI. The directory $data_dir/ARNI arleady exists."
else
    mkdir $data_dir/ARNI;
    wget -P $data_dir/ARNI/ https://zenodo.org/records/6985104/files/Arni_layout.jpg
    wget -P $data_dir/ARNI/ https://zenodo.org/records/6985104/files/Arni_panels_numbers.pdf
    wget -P $data_dir/ARNI/ https://zenodo.org/records/6985104/files/combinations_setup.csv
    wget -P $data_dir/ARNI/ https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_0-5.zip
    wget -P $data_dir/ARNI/ https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_6-15.zip
    wget -P $data_dir/ARNI/ https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_16-25.zip
    wget -P $data_dir/ARNI/ https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_26-35.zip
    wget -P $data_dir/ARNI/ https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_36-45.zip
    wget -P $data_dir/ARNI/ https://zenodo.org/records/6985104/files/IR_Arni_upload_numClosed_46-55.zip
    for file in $data_dir/ARNI/*.zip; do
        n_files=`unzip -l $file | tail -n 1 | xargs echo -n | cut -d' ' -f2`
        unzip $file -d "${file%%.zip}" | tqdm --unit files --unit_scale --total $n_files > /dev/null
        rm $file
    done
fi
