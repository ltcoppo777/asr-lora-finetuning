import os
import sys
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #needed in order to import config because of file levels
from config import config
import csv

def find_all_file_pairs(path):
    pairs = []
    for root, dirs, files in os.walk(path):#goes through subfolders with os.walk **very great feature!
        for filename in files:
            if filename.endswith('.wav'):
                wav_path = os.path.join(root, filename)
                txt_path = os.path.join(root, filename.replace('.wav', '.txt'))
                    
                if os.path.exists(txt_path):
                    pairs.append((wav_path, txt_path))
    return pairs


def normalize_text(txt: str):
    txt = txt.lower().strip()
    txt = txt.strip(".!?") #gets rid of punctuation
    txt = " ".join(txt.split()) #joins the split text back with a single space
    return txt


#below is most of my hard coded folders from before adding the config folder for ease and reproducibility
'''
#path for train folder
#train_path = r"C:\Users\lukec\OneDrive\Desktop\ASR_prep\data\data\train"

#path of test folder
#test_path = r"C:\Users\lukec\OneDrive\Desktop\ASR_prep\data\data\test"

#uses the find_all_file_pairs function to get .wav/.txt pairs

#txt_path = r"C:\Users\lukec\OneDrive\Desktop\ASR_prep\data\data\train\blocks_T2027201\T2027201.txt" <--- my old hardcoded testers, before going through entire file and tested with one
#wav_path = r"C:\Users\lukec\OneDrive\Desktop\ASR_prep\data\data\train\blocks_T2027201\T2027201.wav"

#to make sure the output path exists
#filelist_path = r"C:\Users\lukec\OneDrive\Desktop\ASR_prep\filelists"
#os.makedirs(filelist_path, exist_ok=True)
#opening the CSV file to write'''


os.makedirs(config.FILELIST_PATH, exist_ok=True)


def process_split(split_name, split_dir, pad, volume):

    split_csv_dir = os.path.join(config.FILELIST_PATH, split_name) #appends the split_name onto the file directory
    os.makedirs(split_csv_dir, exist_ok=True) #checks if the file exists, if doesnt it will create it

    csv_path = os.path.join(split_csv_dir, f"{split_name}.csv")#finally creates the CSV for the specific test/train file


    file_pairs = find_all_file_pairs(split_dir)
    file_pairs.sort() #fixes the order of the runs
    print(f"Found {len(file_pairs)} file pairs for '{split_name}'")
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:  #opens csv in write mode
        #**new skill for me** creates a csv writer with its headers
        writer = csv.writer(csvfile)
        writer.writerow(["path", "text", "duration", "source_id"])

        total_segments = 0


        for wav_path, txt_path in file_pairs: 
            '''
            **newly learned code segment** this takes the last part of the file path and splits the file name in order to name all of my csv files accorindgly
            small process breakdown:
            os.path.basename(wav_path) -> "T2027201.wav"
            os.path.splitext(..) ->("T2027201", ".wav")
            [0] -> "T2027201"
            '''
            #makes an ID unique to the folder since T2027201 repeated at first
            rel_folder = os.path.relpath(os.path.dirname(wav_path), start=split_dir)
            rel_folder_clean = rel_folder.replace("\\", "__").replace("/", "__") #replacing separators so theres no cross-os errors mac-windows
            basename = os.path.splitext(os.path.basename(wav_path))[0] #basename gets the trailing name "T200.wav, splittext splits into T200 and .wav, then the [0] makes it just T200"
            source_id = f"{rel_folder_clean}__{basename}"

            out_root = os.path.join(config.PROCESSED_DATA_PATH, split_name) #sets up the output root folder
            os.makedirs(out_root, exist_ok=True)
            out_path = os.path.join(out_root, source_id)
            os.makedirs(out_path, exist_ok=True)

            print(f"\n=== Processing {source_id} ===")
            
            


            segment_count = 1
            
            with open(txt_path, "r", encoding="utf-8") as f: #opens the text file path in read mode
                for raw_text in f:
                    line = raw_text.strip()
                    if not line: #goes past empty lines
                        continue

                    parts = line.split("\t", maxsplit=2) #splits up the current line into 2 floats 1 string
                    
                    start_sec, end_sec, text = parts #assigns variable names to the parts
                    start_sec = float(start_sec) #to keep types
                    end_sec = float(end_sec)
                    processed_wav = os.path.join(out_path, f"{source_id}_SS{start_sec:.1f}.wav") #03d pads with 3 0's for the file name





                    ss = max(0.0, start_sec - pad) #seek start parameter for FFmpeg command for the start reading timestamp
                    to = end_sec + pad #seek to parameter for FFmpeg for the stop reading timestamp
                    speech_duration = end_sec - start_sec #at first I used end_sec - start_sec but since I padded it I have to use the updated start and end times as well


                    #The command that the subprocess will run instead of manually entering in the terminal
                    cmd = [
                        "ffmpeg", "-i", wav_path,
                        "-ss", f"{ss}", "-to", f"{to}",
                        "-ar", "16000", "-ac", "1",
                        "-filter:a", f"volume={volume}",
                        "-y", processed_wav
                    ]

                    print(f"Processing segment {segment_count}: {start_sec}s-{end_sec}s")
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        print(f"✓ Created: {os.path.basename(processed_wav)}")
                        norm_text = normalize_text(text)

                        #adds to CSV
                        relative_path = os.path.relpath(processed_wav, start=config.FILELIST_PATH)
                        writer.writerow([relative_path, norm_text, f"{speech_duration:.2f}", source_id])
                        total_segments +=1
                    else:
                        print(f"✗ Failed segment {segment_count}: {result.stderr[:200]}")
                    
                    segment_count += 1

    print(f"\nProcessing complete!")
    print(f"Processed {len(file_pairs)} files")
    print(f"Created {total_segments} total segments")
    print(f"Filelist saved to: {csv_path}")



process_split("train", config.TRAIN_PATH, config.TRAIN_PAD, config.TRAIN_VOLUME)
process_split("test", config.TEST_PATH, config.TEST_PAD, config.TEST_VOLUME)