from llama_cpp import Llama
import json
import time
import os
from playsound import playsound
from pathlib import Path
import requests

with open('tmp.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Answer内のres内のtextを抽出して変数に代入
res = data['awnser']['res']['text']
print(res)

llm = Llama(model_path="cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-Q8_0.gguf")
prompt=res
prompt="Q: "+prompt+" A: "
output = llm(prompt,max_tokens=100, stop=["Q:", "\n"], echo=True)
speak = output["choices"][0]["text"].split("A: ")[1] if "A: " in output["choices"][0]["text"] else output["choices"][0]["text"]
print(output["choices"][0]["text"])
# クエリの生成
url="http://localhost:50021/audio_query"  # 50021はデフォルトポートなので、変更した場合は変える
params={"text": speak, "speaker": 3}  # 3はずんだもんのノーマルスタイル
timeout = 15  # timeoutは環境によって適切なものを設定する
json_synthesis = requests.post(url, params=params, timeout=timeout)
# 音声合成
response = requests.post(
            "http://localhost:50021/synthesis",
            params=params,
            json=json_synthesis.json()
        )
# 音声を保存
wav = response.content
path = "tmp.wav"  # 保存場所
out = Path(path)
out.write_bytes(wav)
# 再生
playsound("tmp.wav")