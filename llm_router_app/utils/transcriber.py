import whisper
import tempfile
import os

model = whisper.load_model("base")

def transcribe_file(filepath):
    result = model.transcribe(filepath)
    return result["text"]