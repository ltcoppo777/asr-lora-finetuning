import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config import config
import torchaudio, torch
import soundfile as sf
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch import amp
import contextlib

torch.cuda.empty_cache()
MODEL_ID = config.MODEL_ID

def whisper_preprocess(processor, waveform, sr, text): #takes in the raw .wav, the sample rate, and the transcript as well as whispers processor
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)  #convert stereo to mono by averaging channels
    
    if sr != 16000:
        waveform_tensor = torch.from_numpy(waveform)  #numpy --> tensor, was hitting an error since pytorhc expected a tensor
        waveform_tensor = torchaudio.functional.resample(waveform_tensor, orig_freq=sr, new_freq=16000)
        waveform = waveform_tensor.numpy() 
        sr = 16000
    
    features = processor(audio=waveform, sampling_rate=sr, return_attention_mask=True)
    input_features = features["input_features"][0] #removes the batch's dimension so the audio tensor is 2D for Whisper Looks like (1,80,T)...(2,80,T)where T is text, drops the 1 and 2 making (80,T)
    #Had a memory error where I couldnt process multiple batches at once, so I removed it entirely and decided to go one tensor sample at a time
    labels = processor.tokenizer(text, return_tensors="pt").input_ids[0]


    
    return input_features, labels #returns log-mel features, and token labels


def pick_train_rows(csv_path):
    results = []
    counter = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)  # skips header
        for row in r:
            if row and len(row) >= 2 and counter < config.N_SAMPLES:
                path_and_ref_dict = {"relpath": row[0], "ref": row[1]}
                results.append(path_and_ref_dict)
                counter += 1
    return results


def main():
    print("=== Training Whisper with LoRA ===")
    
    #load training CSV (uses config path)
    samples = pick_train_rows(config.TRAIN_CSV)
    print(f"Found {len(samples)} training samples")
    
    # Load processor and model (same as evaluation)
    device = "cuda" if torch.cuda.is_available() else "cpu" #to use my GPU
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)

    model.gradient_checkpointing_enable()   # save activation memory
    model.config.use_cache = False          # disable KV-cache during training

    #refer to the config file to edit
    lora_config = LoraConfig(
        r=config.LORA_RANK,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    model.train()   

    scaler = amp.GradScaler(enabled=(device == "cuda"))
    autocast_ctx = amp.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()

    #this portion will practically process all of the samples and store them in these lists
    all_features = []
    all_labels = []
    print("Preprocessing all samples...")
    for clip in samples:
        relpath = clip["relpath"]
        ref_text = clip["ref"]
        abs_wav = os.path.normpath(os.path.join(config.FILELIST_PATH, relpath))
        
        #print(f"Processing: {ref_text}")
        assert os.path.exists(abs_wav), f"Missing WAV: {abs_wav}"
        
        #loads everything and preprocesses
        wav, sr = sf.read(abs_wav, dtype="float32")
        input_features, label_ids = whisper_preprocess(processor, wav, sr, ref_text)
        
        #add to the lists
        all_features.append(input_features)
        all_labels.append(label_ids)

    print(f"\nCollected {len(all_features)} samples for training")
    print("\nStarting training...")

    for step in range(config.NUM_TRAINING_STEPS):
        #pick one sample to train on (cycle through all samples, usually in same order but can change this)
        sample_idx = step % len(all_features)  #cycle through samples

        #convert single sample to tensors and move to GPU
        features = torch.tensor(all_features[sample_idx], dtype=torch.float16).unsqueeze(0).to(device)  # Single audio sample
        labels = all_labels[sample_idx].clone().unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)  #clears old gradients

        #DEBUG PRINTS
        #print(f"Features shape: {features.shape}")
        #print(f"Labels shape: {labels.shape}")
        #print(f"Labels dtype: {labels.dtype}")
        #print(f"Labels content (first few): {labels.flatten()[:10]}")
        #forward pass - model makes predictions on this ONE sample
        with autocast_ctx:
            outputs = model.base_model.model(
                input_features=features,
                labels=labels
            )
            loss = outputs.loss

        '''
        SAME AS BEFORE - THE MOST IMPORTANT PART
        ---------Steps that they take-----------
        Model gets audio features (single sample now)
        Model gets correct text (single sample now)
        Model makes its own prediction internally
        Model compares its prediction to correct text automatically
        THE LINES BELOW WILL PIGGY BACK OFF OF THIS
        '''

        #backward pass - calculates gradients (direction that the agent should take)
        scaler.scale(loss).backward()   #figures out what caused the errors before
        scaler.step(optimizer)          #adjust the gradients to be less wrong
        scaler.update()

        if step % config.PRINT_EVERY_X_STEPS == 0:  #Print progress
            print(f"Step {step}, Loss: {loss.item():.4f}")

        # per-step to keep VRAM stable
        del features, labels
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTraining complete! Final loss: {loss.item():.4f}")
    model.save_pretrained("./fine_tuned_whisper")


if __name__ == "__main__":
    main()