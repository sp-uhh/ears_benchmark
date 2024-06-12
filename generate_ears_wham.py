import sys
import json
import numpy as np
import pyloudnorm as pyln

from glob import glob
from os import listdir, makedirs
from os.path import join, isdir, exists
from argparse import ArgumentParser
from soundfile import read, write
from tqdm import tqdm


def save_files(target_dir, subset, speaker, id, speech_file, speech_start, speech_end, 
               noise_file, noise_start, mixture, speech, snr_dB, args):
    with open(join(target_dir, f"{subset}.csv"), "a") as text_file:
        text_file.write(f"{id:05},{speaker},{speech_file.split('/')[-1][:-4]},{speech_start},{speech_end},"
            + f"{noise_file.split('/')[-1][:-4]},{noise_start+speech_start},{noise_start+speech_start+len(mixture)},{snr_dB:.1f}\n")
    write(join(target_dir, subset, "noisy", speaker, f"{id:05}_{snr_dB:.1f}dB.wav"), mixture, args.sr, subtype="FLOAT")
    if args.copy_clean:
        write(join(target_dir, subset, "clean", speaker, f"{id:05}.wav"), speech, args.sr, subtype="FLOAT")
    id += 1
    return id

def find_emotion_style(speech_file, emotions_styles=[]):    
    for emo_style in emotions_styles:
        if emo_style.lower() in speech_file.lower():
            return emo_style
    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory which should contain subdirectories EARS and WHAM!48kHz")
    parser.add_argument("--min_snr", type=float, default=-2.5, help="Minimum SNR")
    parser.add_argument("--max_snr", type=float, default=17.5, help="Maximum SNR")
    parser.add_argument("--min_length", type=float, default=4.0, help="Minimum length of speech files in seconds")
    parser.add_argument("--cut_length", type=float, default=10.0, help="Cut long files to this length in seconds")
    parser.add_argument("--copy_clean", action="store_true", help="Copy clean speech files to target directory")
    parser.add_argument("--sr", type=int, default=48000, help="Sampling rate")
    parser.add_argument("--ramp_time_in_ms", type=int, default=10, help="Ramp time in ms")
    parser.add_argument("--max_time_test_set_in_s", type=int, default=29, help="Maximum time in seconds for the test set")
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(42)

    # Organize directories
    speech_dir = join(args.data_dir, "EARS")
    noise_dir = join(args.data_dir, "WHAM48kHz")
    target_dir = join(args.data_dir, "EARS-WHAM")
    assert isdir(speech_dir), f"The directory {speech_dir} does not exist"
    assert isdir(noise_dir), f"The directory {noise_dir} does not exist"

    if exists(join(args.data_dir, target_dir)):
        print(f"[Warning] Abort EARS-WHAM generation script. The directory {join(args.data_dir, target_dir)} already exists.")
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

    # Define emotions and speaking styles
    emotions_styles = [
        "adoration",
        "amazement",
        "amusement",
        "anger",
        "confusion",
        "contentment",
        "cuteness",
        "desire",
        "disappointment",
        "disgust",
        "distress",
        "embarassment",
        "extasy",
        "fast",
        "fear",
        "guilt",
        "highpitch",
        "interest",
        "loud",
        "lowpitch",
        "neutral",
        "pain",
        "pride",
        "realization",
        "relief",
        "regular",
        "sadness",
        "serenity",
        "slow",
        "whisper"
    ]

    # Load noisy speech 
    noise_files = glob(join(noise_dir, "high_res_wham", "audio", "*.wav")) 

    # DSP
    meter = pyln.Meter(args.sr)
    
    # Select speech files for split
    for subset in ["train", "valid"]:
        print(f"Generate {subset} split")
        with open(join(target_dir, f"{subset}.csv"), "w") as text_file:
            text_file.write(f"id,speaker,speech_file,speech_start,speech_end,noise_file,noise_start,noise_end,snr_dB\n")
        speech_files = []
        for speaker in speakers[subset]:  
            speech_files += sorted(glob(join(speech_dir, speaker, "*.wav")))
            makedirs(join(target_dir, subset, "clean", speaker))  
            makedirs(join(target_dir, subset, "noisy", speaker))  
        
        # Remove files of hold out styles
        speech_files = [speech_file for speech_file in speech_files if speech_file.split("/")[-1].split("_")[0] not in hold_out_styles]
        id = 0
        for speech_file in tqdm(speech_files):
            speech, sr = read(speech_file)
            assert sr == args.sr
            speaker = speech_file.split("/")[-2]

            # Only take speech files that are longer than min_length
            if len(speech) < args.min_length*args.sr:
                continue
            
            noise = np.zeros((0,0))
            # Only take noise file that is longer than the speech file
            while noise.shape[0] < speech.shape[0]:
                noise_file = np.random.choice(noise_files)
                noise, sr = read(noise_file, always_2d=True)
            assert sr == args.sr

            # Take random channel if noise file is multi-channel
            channel = np.random.randint(0, noise.shape[1])
            noise = noise[:,channel]

            # Randomly select a part of the noise file
            noise_start = np.random.randint(len(noise)-len(speech)+1)
            noise = noise[noise_start:noise_start+len(speech)]

            # Normalize noise to target SNR
            snr_dB = np.round(np.random.uniform(args.min_snr, args.max_snr), decimals=1)
            loudness_speech = meter.integrated_loudness(speech)
            loudness_noise = meter.integrated_loudness(noise)
            target_loudness = loudness_speech - snr_dB
            delta_loudness = target_loudness - loudness_noise
            gain = np.power(10.0, delta_loudness/20.0)
            noise_scaled = gain * noise
            mixture = speech + noise_scaled

            # Add 1dB to target SNR if mixture is clipping
            while np.max(np.abs(mixture)) >= 1.0:
                snr_dB = snr_dB + 1
                target_loudness = loudness_speech - snr_dB
                delta_loudness = target_loudness - loudness_noise
                gain = np.power(10.0, delta_loudness/20.0)
                noise_scaled = gain * noise
                mixture = speech + noise_scaled

            # Cut long files into pieces
            if len(mixture) >= int((args.cut_length + args.min_length)*args.sr):
                long_mixture = mixture
                long_speech = speech
                num_splits = int((len(long_mixture) - int(args.min_length*args.sr))/int(args.cut_length*args.sr)) + 1
                for i in range(num_splits - 1):
                    speech_start = i*int(args.cut_length*args.sr)
                    speech_end = (i+1)*int(args.cut_length*args.sr)
                    mixture = long_mixture[speech_start:speech_end]
                    speech = long_speech[speech_start:speech_end]
                    id = save_files(target_dir, subset, speaker, id, speech_file, speech_start, speech_end, 
                                    noise_file, noise_start, mixture, speech, snr_dB, args)
                speech_start = (num_splits - 1)*int(args.cut_length*args.sr)
                speech_end = -1
                mixture = long_mixture[speech_start:speech_end]
                speech = long_speech[speech_start:speech_end]
                id = save_files(target_dir, subset, speaker, id, speech_file, speech_start, speech_end, 
                                noise_file, noise_start, mixture, speech, snr_dB, args)
            else:
                speech_start = 0
                speech_end = -1
                id = save_files(target_dir, subset, speaker, id, speech_file, speech_start, speech_end, 
                                noise_file, noise_start, mixture, speech, snr_dB, args)

    # ramps at beginning and end
    ramp_duration = args.ramp_time_in_ms / 1000
    ramp_samples = int(ramp_duration * args.sr)
    ramp = np.linspace(0, 1, ramp_samples)

    # Reset the seed for reproducibility
    np.random.seed(42)

    print("Generate test split")
    with open("test_files.json", "r") as json_file:
        data = json.load(json_file)

    with open(join(target_dir, f"test.csv"), "w") as text_file:
        text_file.write(f"id,speaker,speech_file,speech_start,speech_end,noise_file,noise_start,noise_end,snr_dB\n")

    test_files = []
    for speaker in test_speakers:
        makedirs(join(target_dir, "test", "clean", speaker))  
        makedirs(join(target_dir, "test", "noisy", speaker))  
        speech_files = list(data[speaker].keys())
        for speech_file in speech_files:
            test_files.append(join(speech_dir, speaker, speech_file + ".wav"))

    # Shuffle test files
    np.random.shuffle(test_files)

    # Ensure that the SNR is sampled uniformly for each emotion/style
    number_of_files_per_emotion = 12
    snr_bins = np.linspace(args.min_snr, args.max_snr, number_of_files_per_emotion + 1)
    counter_emotion_style = {x: 0 for x in emotions_styles}

    id = 0
    for test_file in tqdm(test_files):
        speaker = test_file.split("/")[-2]
        speech_file = test_file.split("/")[-1][:-4]

        speech, sr = read(join(speech_dir, speaker, speech_file + ".wav"))
        assert sr == args.sr
        cutting_times = data[speaker][speech_file]

        noise_file = np.random.choice(noise_files)
        noise, sr = read(noise_file, always_2d=True)
        assert sr == args.sr

        # Take random channel if noise file is multi-channel
        channel = np.random.randint(0, noise.shape[1])
        noise = noise[:,channel]

        for cutting_time in cutting_times:
            start = cutting_time[0]
            end = cutting_time[1]
            speech_cut = speech[start:end]

            # Only take speech files that not longer than max_time_test_set_in_s
            if len(speech_cut) > args.max_time_test_set_in_s*args.sr:
                continue

            # Only take noise file that is longer than the speech file
            if noise.shape[0] < speech_cut.shape[0]:
                while noise.shape[0] < speech_cut.shape[0]:
                    noise_file = np.random.choice(noise_files)
                    noise, sr = read(noise_file, always_2d=True)
                assert sr == args.sr

                # Take random channel if noise file is multi-channel
                channel = np.random.randint(0, noise.shape[1])
                noise = noise[:,channel]

            # Randomly select a part of the noise file
            noise_start = np.random.randint(len(noise)-len(speech_cut)+1)
            noise_cut = noise[noise_start:noise_start+len(speech_cut)]

            # Sample SNR uniformly for each emotion/style, else sample uniformly between min_snr and max_snr
            emo_style = find_emotion_style(speech_file, emotions_styles)
            if emo_style is not None:
                index = counter_emotion_style[emo_style] % number_of_files_per_emotion
                min_snr = snr_bins[index]
                max_snr = snr_bins[index+1]
                counter_emotion_style[emo_style] += 1
                snr_dB = np.round(np.random.uniform(min_snr, max_snr), decimals=1)
            else:
                snr_dB = np.round(np.random.uniform(args.min_snr, args.max_snr), decimals=1)

            # Normalize noise to target SNR
            loudness_speech_cut = meter.integrated_loudness(speech_cut)
            loudness_noise = meter.integrated_loudness(noise_cut)
            target_loudness = loudness_speech_cut - snr_dB
            delta_loudness = target_loudness - loudness_noise
            gain = np.power(10.0, delta_loudness/20.0)
            noise_scaled = gain * noise_cut
            mixture = speech_cut + noise_scaled

            # Add 1dB to target SNR if mixture is clipping
            while np.max(np.abs(mixture)) >= 1.0:
                snr_dB = snr_dB + 1
                target_loudness = loudness_speech_cut - snr_dB
                delta_loudness = target_loudness - loudness_noise
                gain = np.power(10.0, delta_loudness/20.0)
                noise_scaled = gain * noise_cut
                mixture = speech_cut + noise_scaled

            # Apply ramps
            mixture[:ramp_samples] = mixture[:ramp_samples] * ramp
            mixture[-ramp_samples:] = mixture[-ramp_samples:] * ramp[::-1]
            speech_cut[:ramp_samples] = speech_cut[:ramp_samples] * ramp
            speech_cut[-ramp_samples:] = speech_cut[-ramp_samples:] * ramp[::-1]

            id = save_files(target_dir, "test", speaker, id, test_file, start, end, 
                            noise_file, noise_start, mixture, speech_cut, snr_dB, args)
