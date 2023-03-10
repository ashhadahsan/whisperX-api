import whisperx as whisper

from deep_translator import GoogleTranslator
import os
from whisperx.utils import write_vtt, write_srt, write_ass, write_tsv, write_txt
import warnings

warnings.simplefilter("ignore")


async def detect_language(filename, model):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(file=filename)
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    return {"detected_language": max(probs, key=probs.get)}


async def translate_to_english(transcription, json=False):
    if json:
        for text in transcription:
            text["text"] = GoogleTranslator(source="auto", target="en").translate(
                text["text"]
            )
    else:

        for text in transcription["segments"]:
            text["text"] = GoogleTranslator(source="auto", target="en").translate(
                text["text"]
            )
    return transcription


async def write(filename, dtype, result_aligned):

    if dtype == "vtt":
        with open(
            os.path.join(".", os.path.splitext(filename)[0] + ".vtt"), "w",encoding="utf-8"
        ) as vtt:
            write_vtt(result_aligned["segments"], file=vtt)
    if dtype == "srt":
        with open(
            os.path.join(".", os.path.splitext(filename)[0] + ".srt"),
            "w",
            encoding="utf-8",
        ) as srt:
            write_srt(result_aligned["segments"], file=srt)
    if dtype == "ass":
        with open(
            os.path.join(".", os.path.splitext(filename)[0] + ".ass"), "w",encoding="utf-8"
        ) as ass:
            write_ass(result_aligned["segments"], file=ass)
    if dtype == "tsv":
        with open(
            os.path.join(".", os.path.splitext(filename)[0] + ".tsv"), "w",encoding="utf-8"
        ) as tsv:
            write_tsv(result_aligned["segments"], file=tsv)
    if dtype == "plain text":
        print("here")
        print(filename)
        with open(
            os.path.join(".", os.path.splitext(filename)[0] + ".txt"), "w",encoding="utf-8"
        ) as txt:
            write_txt(result_aligned["segments"], file=txt)


async def read(filename, transc):
    if transc == "plain text":
        transc = "txt"
    filename = filename.split(".")[0]
    print(filename)
    with open(f"{filename}.{transc}", encoding="utf-8", errors="ignore") as f:
        content = f.readlines()
    content = " ".join(z for z in content)
    return content, f"{filename}.{transc}"


from constants import language_dict


async def get_key(val):
    for key, value in language_dict.items():
        if val == value:
            return key
    return "Key not found"
