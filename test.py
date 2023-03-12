import requests
import json
url = "http://localhost:8000/whisperX/"
# open the JSON file to upload
#aligned json is optional and must only be passed if u already have the json file

with open("../notebook/fold.json", "r") as input_file:
    data = json.load(input_file)
files = {'audio': ('../audio/fold.wav', open('../audio/fold.wav', 'rb'), 'audio/wav'),
         'aligned_json': ("../notebook/fold.json", json.dumps(data), "application/json")}
data = {'transcription': 'srt', 'model_name': 'tiny','translate':False,
        "device":"cpu",
        "language":"en","patience":1.0,"temperature":1.0,
        "condition_on_previous/_text":False,"temperature_increment_on_fallback":0.2,
        "compression_ratio_threshold":2.4,"logprob_threshold":-1.0,"no_speech_threshold":0.6
        }

response = requests.post(url, files=files, data=data)

response.json()
