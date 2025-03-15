import torch
import flet as ft
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyaudio
import numpy as np
import json
import subprocess
import sys

def main(page: ft.Page):
    page.title = "Speech Recognition App"
    
    # Set up models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    
    status_text = ft.Text("Loading model... This may take a moment.")
    page.add(status_text)
    page.update()
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True
    )
    
    status_text.value = "Model loaded. Ready to record."
    page.update()
    
    result_text = ft.Text("Recognition result will appear here")
    
    def record_audio(duration=5, sample_rate=16000):
        # PyAudio setup
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                       channels=1,
                       rate=sample_rate,
                       input=True,
                       frames_per_buffer=1024)
        
        status_text.value = f"Recording... {duration} seconds"
        page.update()
        
        frames = []
        
        # Record for specified duration
        for i in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))
        
        status_text.value = "Recording complete. Processing..."
        page.update()
        
        # Close stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Combine audio data
        audio_data = np.concatenate(frames, axis=0)
        return {"array": audio_data, "sampling_rate": sample_rate}
    
    def process_recognition_results(result):
        try:
            # Read JSON file
            try:
                with open('tmp.json', 'r', encoding="utf-8") as f:
                    dict_json = json.load(f)
                
                # Check JSON structure and update correctly
                if isinstance(dict_json, dict):
                    if 'awnser' not in dict_json:
                        dict_json['awnser'] = {}
                    if not isinstance(dict_json['awnser'], dict):
                        dict_json['awnser'] = {}
                    dict_json['awnser']['res'] = result
                else:
                    dict_json = {'awnser': {'res': result}}
            except (FileNotFoundError, json.JSONDecodeError):
                dict_json = {'awnser': {'res': result}}
            
            # Write to JSON file with correct Unicode handling
            with open('tmp.json', 'w', encoding="utf-8") as new_json:
                json.dump(dict_json, new_json, ensure_ascii=False, indent=4)
                
        except Exception as e:
            status_text.value = f"Error occurred: {e}"
            page.update()
    
    def ninsiki(e):
        # Record audio
        audio_input = record_audio()
        
        # Speech recognition
        result = pipe(audio_input)
        result_text.value = f"Recognition result: {result['text']}"
        page.update()
        
        # Process results
        process_recognition_results(result)
        
        # Run main2.py
        status_text.value = "Running main2.py..."
        page.update()
        
        try:
            subprocess.run(["python", "main2.py"])
            status_text.value = "main2.py execution completed."
        except Exception as e:
            status_text.value = f"Error running main2.py: {e}"
        
        page.update()
    
    # UI Elements
    page.add(
        ft.Column([
            status_text,
            ft.ElevatedButton("Record for 5 seconds", on_click=ninsiki),
            result_text,
            ft.ElevatedButton("Exit", on_click=lambda e: sys.exit(0))
        ])
    )

ft.app(main)