import sys
import sofa
import mat73
import numpy as np
import pyloudnorm as pyln

import json
from glob import glob
from os import listdir, makedirs
from os.path import join, isdir, exists
from argparse import ArgumentParser
from soundfile import read, write
from tqdm import tqdm
from scipy.signal import convolve
from scipy import stats
from librosa import resample


def save_files(target_dir, subset, speaker, id, speech_file, speech_start, speech_end, rir_file, channel, 
               gain, rt60, mixture, speech, args):
    with open(join(target_dir, f"{subset}.csv"), "a") as text_file:
        text_file.write(f"{id:05},{speaker},{speech_file.split('/')[-1][:-4]},{speech_start},{speech_end},"
            + f"{rir_file.replace(args.data_dir, '')},{channel},{gain},{rt60:.2f}\n")
    write(join(target_dir, subset, "reverberant", speaker, f"{id:05}_{rt60:.2f}.wav"), mixture, args.sr, subtype="FLOAT")
    if args.copy_clean:
        write(join(target_dir, subset, "clean", speaker, f"{id:05}.wav"), speech, args.sr, subtype="FLOAT")
    id += 1
    return id

def calc_rt60(h, sr=480000, rt='t30'): 
    """
    RT60 measurement routine acording to Schroeder's method [1].

    [1] M. R. Schroeder, "New Method of Measuring Reverberation Time," J. Acoust. Soc. Am., vol. 37, no. 3, pp. 409-412, Mar. 1968.

    Adapted from https://github.com/python-acoustics/python-acoustics/blob/99d79206159b822ea2f4e9d27c8b2fbfeb704d38/acoustics/room.py#L156
    """
    rt = rt.lower()
    if rt == 't30':
        init = -5.0
        end = -35.0
        factor = 2.0
    elif rt == 't20':
        init = -5.0
        end = -25.0
        factor = 3.0
    elif rt == 't10':
        init = -5.0
        end = -15.0
        factor = 6.0
    elif rt == 'edt':
        init = 0.0
        end = -10.0
        factor = 6.0

    h_abs = np.abs(h) / np.max(np.abs(h))

    # Schroeder integration
    sch = np.cumsum(h_abs[::-1]**2)[::-1]
    sch_db = 10.0 * np.log10(sch / np.max(sch)+1e-20)

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / sr
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time (T30, T20, T10 or EDT)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)
    return t60


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help='Path to data directory which should contain subdirectories EARS and WHAM!48kHz')
    parser.add_argument("--min_length", type=float, default=4.0, help='Minimum length of speech files in seconds')
    parser.add_argument("--cut_length", type=float, default=10.0, help='Cut long files to this length in seconds')
    parser.add_argument("--copy_clean", action='store_true', help='Copy clean speech files to target directory')
    parser.add_argument("--sr", type=int, default=48000, help='Sampling rate')
    parser.add_argument("--ramp_time_in_ms", type=int, default=10, help="Ramp time in ms")
    parser.add_argument("--max_rt60", type=float, default=2.0, help="Maximum RT60 in seconds")
    parser.add_argument("--max_time_test_set_in_s", type=int, default=29, help="Maximum time in seconds for the test set")
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(42)

    # Organize directories
    speech_dir = join(args.data_dir, "EARS")
    target_dir = join(args.data_dir, "EARS-Reverb_seed_42")
    assert isdir(speech_dir), f"The directory {speech_dir} does not exist"

    if exists(target_dir):
        print(f"[Warning] Abort EARS-Reverb generation script. The directory {join(args.data_dir, target_dir)} already exists.")
        sys.exit()
    else:
        makedirs(target_dir)

    all_speakers = sorted(listdir(speech_dir))
    # Define training split
    valid_speakers = ["p100", "p101"] 
    test_speakers = ["p102", "p103", "p104", "p105", "p106", "p107"]
    
    speakers = {
        "train": [s for s in all_speakers if s not in valid_speakers + test_speakers],
        "valid": valid_speakers, 
        "test": test_speakers
        }
    
    # Hold out speaking styles 
    hold_out_styles = ["interjection", "melodic", "nonverbal", "vegetative"]

    rir_files = []

    # ACE-Challenge dataset
    dir = join(args.data_dir, "ACE-Challenge")
    names = ["Chromebook", "Crucif", "EM32", "Lin8Ch", "Mobile", "Single"]
    for name in names:
        rir_files += sorted(glob(join(dir, name, "**", "*RIR.wav"), recursive=True))

    # AIR dataset
    dir = join(args.data_dir, "AIR", "AIR_1_4", "AIR_wav_files")
    rir_files += sorted(glob(join(dir, "*.wav")))

    # ARNI dataset
    dir = join(args.data_dir, "ARNI")
    all_arni_files = sorted(glob(join(dir, "**", "*.wav"), recursive=True))
    # remove file numClosed_26-35/IR_numClosed_28_numComb_2743_mic_4_sweep_5.wav because it is corrupted
    all_arni_files = [file for file in all_arni_files if "numClosed_26-35/IR_numClosed_28_numComb_2743_mic_4_sweep_5.wav" not in file]
    rir_files += sorted(list(np.random.choice(all_arni_files, size=1000, replace=False))) # take 1000 of 132037 RIRs

    # BRUDEX dataset
    dir = join(args.data_dir, "BRUDEX")
    rir_files += sorted(glob(join(dir, "rir", "**", "*.mat"), recursive=True))

    # dEchorate dataset
    dir = join(args.data_dir, "dEchorate", "sofa")
    rir_files += sorted(glob(join(dir, "**", "*.sofa"), recursive=True))

    # DetmoldSRIR dataset
    dir = join(args.data_dir, "DetmoldSRIR")
    rir_files += sorted(glob(join(dir, "SetA_SingleSources", "Data", "**", "*.wav"), recursive=True))

    # Palimpsest dataset
    dir = join(args.data_dir, "Palimpsest")
    rir_files += sorted(glob(join(dir, "**", "*.wav"), recursive=True))

    meter = pyln.Meter(args.sr)
    
    # # Select speech files for split
    # for subset in ["train", "valid"]:
    #     print(f"Generate {subset} split")
    #     with open(join(target_dir, f"{subset}.csv"), "w") as text_file:
    #         text_file.write(f"id,speaker,speech_file,speech_start,speech_end,rir_file,channel,gain,rt60\n")
    #     speech_files = []
    #     for speaker in speakers[subset]:  
    #         speech_files += sorted(glob(join(speech_dir, speaker, "*.wav")))
    #         if args.copy_clean:
    #             makedirs(join(target_dir, subset, "clean", speaker))  
    #         makedirs(join(target_dir, subset, "reverberant", speaker))  
        
    #     # Remove files of hold out styles
    #     speech_files = [speech_file for speech_file in speech_files if speech_file.split("/")[-1].split("_")[0] not in hold_out_styles]
    #     id = 0
    #     for speech_file in tqdm(speech_files):
    #         speech, sr = read(speech_file)
    #         assert sr == args.sr
    #         speaker = speech_file.split("/")[-2]

    #         # Only take speech files that are longer than min_length
    #         if len(speech) < args.min_length*args.sr:
    #             continue

    #         # Sample RIRs until RT60 is below max_rt60 and pre_samples are below max_pre_samples
    #         rt60 = np.inf
    #         while rt60 > args.max_rt60:
    #             rir_file = np.random.choice(rir_files)

    #             if "ARNI" in rir_file:
    #                 rir, sr = read(rir_file, always_2d=True)
    #                 # Take random channel if file is multi-channel
    #                 channel = np.random.randint(0, rir.shape[1])
    #                 rir = rir[:,channel]
    #                 assert sr == 44100, f"Sampling rate of {rir_file} is {sr}"
    #                 rir = resample(rir, orig_sr=sr, target_sr=args.sr)
    #                 sr = args.sr
    #             elif rir_file.endswith(".wav"):
    #                 rir, sr = read(rir_file, always_2d=True)
    #                 # Take random channel if file is multi-channel
    #                 channel = np.random.randint(0, rir.shape[1])
    #                 rir = rir[:,channel]
    #             elif rir_file.endswith(".sofa"):
    #                 hrtf = sofa.Database.open(rir_file)
    #                 rir = hrtf.Data.IR.get_values()
    #                 channel = np.random.randint(0, rir.shape[1])
    #                 rir = rir[0,channel,:]
    #                 sr = hrtf.Data.SamplingRate.get_values().item()
    #             elif rir_file.endswith(".mat"):
    #                 rir = mat73.loadmat(rir_file)
    #                 sr = rir["fs"].item()
    #                 rir = rir["data"]
    #                 channel = np.random.randint(0, rir.shape[1])
    #                 rir = rir[:,channel]
    #             else:
    #                 raise ValueError(f"Unknown file format: {rir_file}")

    #             assert sr == args.sr, f"Sampling rate of {rir_file} is {sr}"

    #             # Cut RIR to get direct path at the beginning
    #             max_index = np.argmax(np.abs(rir))
    #             rir = rir[max_index:]

    #             # Normalize RIRs in range [0.1, 0.7]
    #             if np.max(np.abs(rir)) < 0.1:
    #                 rir = 0.1 * rir / np.max(np.abs(rir))
    #             elif np.max(np.abs(rir)) > 0.7:
    #                 rir = 0.7 * rir / np.max(np.abs(rir))

    #             rt60 = calc_rt60(rir, sr=sr)

    #             mixture = convolve(speech, rir)[:len(speech)]

    #             # normalize mixture
    #             loudness_speech = meter.integrated_loudness(speech)
    #             loudness_mixture = meter.integrated_loudness(mixture)
    #             delta_loudness = loudness_speech - loudness_mixture
    #             gain = np.power(10.0, delta_loudness/20.0)
    #             # if gain is inf sample again
    #             if np.isinf(gain):
    #                 rt60 = np.inf
    #             mixture = gain * mixture

    #         if np.max(np.abs(mixture)) > 1.0:
    #             mixture = mixture / np.max(np.abs(mixture))

    #         # Cut long files into pieces
    #         if len(mixture) >= int((args.cut_length + args.min_length)*args.sr):
    #             long_mixture = mixture
    #             long_speech = speech
    #             num_splits = int((len(long_mixture) - int(args.min_length*args.sr))/int(args.cut_length*args.sr)) + 1
    #             for i in range(num_splits - 1):
    #                 speech_start = i*int(args.cut_length*args.sr)
    #                 speech_end = (i+1)*int(args.cut_length*args.sr)
    #                 mixture = long_mixture[speech_start:speech_end]
    #                 speech = long_speech[speech_start:speech_end]
    #                 id = save_files(target_dir, subset, speaker, id, speech_file, speech_start, speech_end, rir_file,
    #                                 channel, gain, rt60, mixture, speech, args)
    #             speech_start = (num_splits - 1)*int(args.cut_length*args.sr)
    #             speech_end = -1
    #             mixture = long_mixture[speech_start:speech_end]
    #             speech = long_speech[speech_start:speech_end]
    #             id = save_files(target_dir, subset, speaker, id, speech_file, speech_start, speech_end, rir_file,
    #                             channel, gain, rt60, mixture, speech, args)
    #         else:
    #             speech_start = 0
    #             speech_end = -1
    #             id = save_files(target_dir, subset, speaker, id, speech_file, speech_start, speech_end, rir_file,
    #                             channel, gain, rt60, mixture, speech, args)
    
    # ramps at beginning and end
    ramp_duration = args.ramp_time_in_ms / 1000
    ramp_samples = int(ramp_duration * args.sr)
    ramp = np.linspace(0, 1, ramp_samples)
                
    print("Generate test split")
    with open("test_files.json", "r") as json_file:
        data = json.load(json_file)

    with open(join(target_dir, f"test.csv"), "w") as text_file:
        text_file.write(f"id,speaker,speech_file,speech_start,speech_end,rir_file,channel,gain,rt60\n")

    test_speakers = list(data.keys())

    test_files = []
    for speaker in test_speakers:
        makedirs(join(target_dir, "test", "clean", speaker))  
        makedirs(join(target_dir, "test", "reverberant", speaker))  
        speech_files = list(data[speaker].keys())
        for speech_file in speech_files:
            test_files.append(join(speech_dir, speaker, speech_file + ".wav"))

    # Reproducibility
    np.random.seed(42)
    np.random.shuffle(test_files)

    id = 0
    for test_file in tqdm(test_files):
        speaker = test_file.split("/")[-2]
        speech_file = test_file.split("/")[-1][:-4]

        speech, sr = read(join(speech_dir, speaker, speech_file + ".wav"))
        assert sr == args.sr
        cutting_times = data[speaker][speech_file]

        for cutting_time in cutting_times:
            start = cutting_time[0]
            end = cutting_time[1]
            speech_cut = speech[start:end]

            # Only take speech files that not longer than max_time_test_set_in_s
            if len(speech_cut) > args.max_time_test_set_in_s*args.sr:
                continue
            
            # Sample RIRs until RT60 is below max_rt60 and pre_samples are below max_pre_samples
            rt60 = np.inf
            while rt60 > args.max_rt60:
                rir_file = np.random.choice(rir_files)

                if "ARNI" in rir_file:
                    rir, sr = read(rir_file, always_2d=True)
                    # Take random channel if file is multi-channel
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[:,channel]
                    assert sr == 44100, f"Sampling rate of {rir_file} is {sr}"
                    rir = resample(rir, orig_sr=sr, target_sr=args.sr)
                    sr = args.sr
                elif rir_file.endswith(".wav"):
                    rir, sr = read(rir_file, always_2d=True)
                    # Take random channel if file is multi-channel
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[:,channel]
                elif rir_file.endswith(".sofa"):
                    hrtf = sofa.Database.open(rir_file)
                    rir = hrtf.Data.IR.get_values()
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[0,channel,:]
                    sr = hrtf.Data.SamplingRate.get_values().item()
                elif rir_file.endswith(".mat"):
                    rir = mat73.loadmat(rir_file)
                    sr = rir["fs"].item()
                    rir = rir["data"]
                    channel = np.random.randint(0, rir.shape[1])
                    rir = rir[:,channel]
                else:
                    raise ValueError(f"Unknown file format: {rir_file}")

                assert sr == args.sr, f"Sampling rate of {rir_file} is {sr}"

                # Cut RIR to get direct path at the beginning
                max_index = np.argmax(np.abs(rir))
                rir = rir[max_index:]

                # Normalize RIRs in range [0.1, 0.7]
                if np.max(np.abs(rir)) < 0.1:
                    rir = 0.1 * rir / np.max(np.abs(rir))
                elif np.max(np.abs(rir)) > 0.7:
                    rir = 0.7 * rir / np.max(np.abs(rir))

                rt60 = calc_rt60(rir, sr=sr)

                mixture = convolve(speech_cut, rir)[:len(speech_cut)]
            
                # normalize mixture
                loudness_speech = meter.integrated_loudness(speech_cut)
                loudness_mixture = meter.integrated_loudness(mixture)
                delta_loudness = loudness_speech - loudness_mixture
                gain = np.power(10.0, delta_loudness/20.0)
                mixture = gain * mixture
                # if gain is inf sample again 
                if np.isinf(gain):
                    rt60 = np.inf

            if np.max(np.abs(mixture)) > 1.0:
                mixture = mixture / np.max(np.abs(mixture))

            # Apply ramps
            mixture[:ramp_samples] = mixture[:ramp_samples] * ramp
            mixture[-ramp_samples:] = mixture[-ramp_samples:] * ramp[::-1]
            speech_cut[:ramp_samples] = speech_cut[:ramp_samples] * ramp
            speech_cut[-ramp_samples:] = speech_cut[-ramp_samples:] * ramp[::-1]

            id = save_files(target_dir, "test", speaker, id, test_file, start, end, rir_file,
                                channel, gain, rt60, mixture, speech_cut, args)
