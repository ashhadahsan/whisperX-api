from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from constants import all_languages, WHISPER_MODELS
from utils import *
from typing import List, Optional
import numpy as np
import whisperx as whisper
from aiofiles.os import remove
import json

app = FastAPI(timeout=60)
# Mount the Swagger UI at /docs
app.mount("/docs", StaticFiles(directory="."), name="docs")

# Mount the ReDoc at /redoc
app.mount("/redoc", StaticFiles(directory="."), name="redoc")

import uuid


@app.post("/whisperX/")
async def upload_file(
    audio: UploadFile = File(...),
    aligned_json: UploadFile = File(None),
    device: str = Form("cpu"),
    model_name: str = Form("base"),
    transcription: str = Form("plain text"),
    translate: bool = Form(False),
    language: str = Form(""),
    patience: float = Form(1.0),
    temperature: float = Form(1.0),
    suppress_tokens: List[str] = Form(["-1"]),
    initial_prompt: str = Form(""),
    condition_on_previous_text: bool = Form(False),
    temperature_increment_on_fallback: float = Form(0.20),
    compression_ratio_threshold: float = Form(2.4),
    logprob_threshold: float = Form(-1.0),
    no_speech_threshold: float = Form(0.6),
):

    if device.lower() != "CPU".lower() and device.lower() != "GPU".lower():
        return {"error": """Invalid 'device' input. Must be either 'CPU' or 'GPU'."""}
    if model_name not in WHISPER_MODELS:
        return {
            "error": f"""Invalid 'model_name' input. Must be either {WHISPER_MODELS}."""
        }

    if transcription.lower() not in ["plain text", "srt", "vtt", "ass", "tsv"]:
        return {
            "error": """Invalid 'transcripton' input. Must be ["plain text", "srt", "vtt", "ass", "tsv"]  ."""
        }
    if language not in all_languages:
        return {"error": """This language is not currently supported ."""}

    if temperature_increment_on_fallback is not None:
        temperature = np.arange(
            temperature, 1.0 + 1e-6, temperature_increment_on_fallback
        )

    elif temperature_increment_on_fallback is None:
        temperature = [temperature]
    try:
        if len(temperature) == 0:
            return {"error": "Invalid value for temperature"}
    except:
        pass

    name = str(uuid.uuid1())
    with open(f"{name}.wav", "wb") as out_file:
        content = await audio.read()  # async read
        out_file.write(content)  # async write

    if language == "":

        model = whisper.load_model(model_name)
        detection = detect_language(f"{name}.wav", model)
        language = detection.get("detected_language")
        del model
    if len(language) > 2:
        language = get_key(language)

    decode = {"suppress_tokens": suppress_tokens, "beam_size": 5.0}
    model = whisper.load_model(model_name)
    if aligned_json is None:

        result = model.transcribe(
            f"{name}.wav",
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
        )

        if translate:
            result = translate_to_english(result, json=False)
        model_a, metadata = whisper.load_align_model(
            language_code=result["language"], device=device
        )
        result_aligned = whisper.align(
            result["segments"],
            model_a,
            metadata,
            f"{name}.wav",
            device=device,
        )
        await write(f"{name}.wav", dtype=transcription, result_aligned=result_aligned)
        trans_text, fname = await read(f"{name}.wav", transcription)
        await remove(f"{name}.wav")
        await remove(fname)
        char_segments = []
        word_segments = []

        for x in range(len(result_aligned["segments"])):
            word_segments.append(
                {
                    "word-segments": result_aligned["segments"][x]["word-segments"]
                    .fillna("")
                    .to_dict(orient="records")
                }
            )
            char_segments.append(
                {
                    "char-segments": result_aligned["segments"][x]["char-segments"]
                    .fillna("")
                    .to_dict(orient="records")
                }
            )

        for x in range(len(result_aligned["segments"])):

            result_aligned["segments"][x]["word-segments"] = word_segments[x]
            result_aligned["segments"][x]["char-segments"] = char_segments[x]
        del model, model_a, metadata

        return {
            "segments_before_alignment": result["segments"],
            "segments_after_alignment": result_aligned["segments"],
            "word_segments_after_alignment": result_aligned["word_segments"],
            "detected language": language_dict.get(language),
            "transcription": trans_text,
        }
    if aligned_json is not None:

        json_filname = str(uuid.uuid1())
        # print(aligned_json)
        contents = await aligned_json.read()  # read contents of the uploaded file
        cont = json.loads(contents)

        # with open(f"{json_filname}.json", "w", encoding="utf-8") as new_file:
        #     new_file.write(contents)

        # with open(f"{json_filname}.json", "r", encoding="utf-8") as f:
        #     cont = json.load(f)

        model_a, metadata = whisper.load_align_model(
            language_code=language, device=device
        )
        result_aligned = whisper.align(
            cont,
            model_a,
            metadata,
            f"{name}.wav",
            device=device,
        )
        await write(
            f"{name}.wav",
            dtype=transcription,
            result_aligned=result_aligned,
        )
        trans_text, fname = await read(f"{name}.wav", transcription)
        await remove(f"{name}.wav")
        await remove(fname)
        char_segments = []
        word_segments = []

        for x in range(len(result_aligned["segments"])):
            word_segments.append(
                {
                    "word-segments": result_aligned["segments"][x]["word-segments"]
                    .fillna("")
                    .to_dict(orient="records")
                }
            )
            char_segments.append(
                {
                    "char-segments": result_aligned["segments"][x]["char-segments"]
                    .fillna("")
                    .to_dict(orient="records")
                }
            )

        for x in range(len(result_aligned["segments"])):

            result_aligned["segments"][x]["word-segments"] = word_segments[x]
            result_aligned["segments"][x]["char-segments"] = char_segments[x]
        del model, model_a, metadata

        return {
            "segments_before_alignment": cont,
            "segments_after_alignment": result_aligned["segments"],
            "word_segments_after_alignment": result_aligned["word_segments"],
            "detected language": language_dict.get(language),
            "transcription": trans_text,
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
