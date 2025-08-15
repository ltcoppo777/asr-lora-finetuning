import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config import config
import torchaudio, torch
import soundfile as sf
from jiwer import wer
from peft import PeftModel

def normalize_text_eval(txt: str):
    txt = txt.lower().strip()
    txt = txt.strip(".!?")            
    return txt



def pick_all_rows(csv_path):
    results = []
    counter = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)  #skips the header row
        for row in r:
            if row and len(row) >= 2 and counter < config.N_SAMPLES:
                path_and_ref_dict = {"relpath": row[0], "ref": row[1]}
                results.append(path_and_ref_dict)
                counter+=1
    return results

      
def main():
    test_csv = config.TEST_CSV
    samples = pick_all_rows(test_csv)

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    
    if config.EVAL_MODEL_PATH.startswith("./"):
        
        #loads the fine tuned model
        base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        model = PeftModel.from_pretrained(base_model, config.EVAL_MODEL_PATH).to(device).eval()
    else:
        #loads base model
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device).eval()

    print(f"Evaluating model: {config.EVAL_MODEL_PATH}")

    all_wer_scores = []
    for clips in samples:
        relpath = clips["relpath"]
        ref_transcript = clips["ref"]
        abs_wav = os.path.normpath(os.path.join(config.FILELIST_PATH, relpath))

    
        print("CSV path (relative):", relpath)
        print("Absolute WAV path  :", abs_wav)
        print("Reference transcript:", ref_transcript)
        
        assert os.path.exists(abs_wav), f"Missing WAV: {abs_wav}"
        

        waveform, sr = sf.read(abs_wav, dtype="float32")
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000


        inputs = processor(audio=waveform, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}


        with torch.inference_mode():
            pred_ids = model.generate(**inputs, language="en", task="transcribe") #runs the decoder to produce token IDs for the transcription
            #The auto language detection feature gave me an error so I had to hard code it to english per Whisper's policy

            hyp = processor.batch_decode(pred_ids, skip_special_tokens=True)[0] #decodes back to human readable string from the tokens
            print("MODEL:", hyp)

        ref_norm = normalize_text_eval(ref_transcript)
        hyp_norm = normalize_text_eval(hyp)              
        clip_wer = wer(ref_norm, hyp_norm)
        all_wer_scores.append(clip_wer)
        print("Total WER for clip: ", clip_wer)

    average_wer = sum(all_wer_scores) / len(all_wer_scores)
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS for {config.EVAL_MODEL_PATH}")
    print(f"{'='*50}")
    print(f"Samples evaluated: {len(samples)}")
    print(f"Average WER: {average_wer:.4f} ({average_wer*100:.2f}%)")
    print(f"Best WER: {min(all_wer_scores):.4f}")
    print(f"Worst WER: {max(all_wer_scores):.4f}")


if __name__ == "__main__":
    main()