import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyaudio
import wave
import numpy as np
import keyboard
import json
from pathlib import Path
import time
import subprocess
import sys

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"
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
 
def record_audio(duration=5, sample_rate=16000):
    # PyAudioの設定
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                   channels=1,
                   rate=sample_rate,
                   input=True,
                   frames_per_buffer=1024)
   
    print(f"録音開始... {duration}秒間")
    frames = []
   
    # 指定された時間だけ録音
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.float32))
   
    print("録音終了")
   
    # ストリームを閉じる
    stream.stop_stream()
    stream.close()
    p.terminate()
   
    # 音声データを結合
    audio_data = np.concatenate(frames, axis=0)
    return {"array": audio_data, "sampling_rate": sample_rate}

print("スペースキーを押すと5秒間の録音を開始します。'q'キーで終了します。")

while True:
    if keyboard.is_pressed('q'):
        print("プログラムを終了します")
        break
       
    if keyboard.is_pressed('space'):
        # 録音
        audio_input = record_audio()
        # 音声認識
        result = pipe(audio_input)
        print("認識結果:", result["text"])
        
        try:
            # JSONファイルを読み込む
            with open('tmp.json', 'r', encoding="utf-8") as f:
                dict_json = json.load(f)
                
            # JSONの構造を確認して正しく更新
            if isinstance(dict_json, dict):
                if 'awnser' not in dict_json:
                    dict_json['awnser'] = {}
                if not isinstance(dict_json['awnser'], dict):
                    dict_json['awnser'] = {}
                dict_json['awnser']['res'] = result
            else:
                dict_json = {'awnser': {'res': result}}
            
            # 日本語をUnicodeエスケープせずに正しく書き込む
            with open('tmp.json', 'w', encoding="utf-8") as new_json:
                json.dump(dict_json, new_json, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            # JSONファイルが存在しない場合は新規作成
            with open('tmp.json', 'w', encoding="utf-8") as new_json:
                json.dump({'awnser': {'res': result}}, new_json, ensure_ascii=False, indent=4)
        
        # スペースキーが離されるまで待機
        while keyboard.is_pressed('space'):
            time.sleep(0.1)
        
        # main2.pyを実行
        print("main2.pyを実行します...")
        subprocess.run(["python", "main2.py"])
        print("main2.pyの実行が完了しました")
        
        # プログラムを終了
        print("プログラムを終了します")
        sys.exit(0) 