from fastapi import FastAPI, UploadFile, File
import subprocess
import uuid
import os
from pydantic import BaseModel
import re
import httpx
from fastapi import HTTPException

class ReflectRequest(BaseModel):
    text: str
class SpeakRequest(BaseModel):
    text: str
    voice: str = "lessac"  # Options: lessac, amy, ljspeech


app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
WHISPER_BIN = "whisper-cli"
MODEL_PATH = "/Users/poonamligade/Documents/poonam/whisper.cpp/models/ggml-base.en.bin"
LLAMA_MODEL = "/Users/poonamligade/llm-models/qwen2.5-0.5b-instruct.gguf"
# LLAMA_BIN = "llama-cli"

LLAMA_SERVER_URL = "http://localhost:8080/completion"
PIPER_MODELS = {
    "lessac": os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx"),
    "amy": os.path.expanduser("~/piper-voices/en_US-amy-medium.onnx"),
    "ljspeech": os.path.expanduser("~/piper-voices/en_US-ljspeech-medium.onnx")
}

# REFLECTION_SYSTEM_PROMPT = """
# You are a calm journaling companion. Ask one gentle reflective question about my day.
# """
os.makedirs("audio", exist_ok=True)
os.makedirs("journals", exist_ok=True)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:8]
    raw_path = f"audio/chunk.wav"
    fixed_path = f"audio/chunk_16k.wav"
    journal_path = f"journals/chunk.txt"

    # Save uploaded audio
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    # Convert to 16 kHz
    subprocess.run(
        ["sox", raw_path, "-r", "16000", fixed_path],
        check=True
    )

    # Run whisper
    result = subprocess.run(
        [
            "whisper-cli",
            "-m", MODEL_PATH,
            "-f", fixed_path,
            "--no-timestamps",
            "-l", "en"
        ],
        capture_output=True,
        text=True,
        check=True
    )

    transcript = result.stdout.strip()

    # Save transcript
    with open(journal_path, "w") as f:
        f.write(transcript)

    return {
        "session_id": session_id,
        "text": transcript
    }

# @app.post("/reflect")
# def reflect(req: ReflectRequest):
#     text = req.text
#     prompt = f"""
#     You are a calm journaling companion.
#     Here is my journal entry:
#     "{text}"
#     Ask one reflective follow-up question.
#     """

#     result = subprocess.run(
#         [
#             "llama-cli",
#             "-m", LLAMA_MODEL,
#             "--ctx-size", "2048",
#             "-n", "64",          # max tokens
#             "--temp", "0.7",
#             "--no-display-prompt",
#             "--log-disable", 
#             "--simple-io",
#             "-p", prompt
#         ],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,   # capture errors safely
#         # capture_output=True,
#         text=True,
#         #check=True
#     )
#     if result.returncode != 0:
#         return {
#             "error": "llama-cli failed",
#             "stderr": result.stderr
#         }

#     raw = result.stdout.strip()
#     print(raw)
#     # ✅ Extract the LAST question sentence (robust)
#     questions = re.findall(r'([A-Z][^?.!]*\?)', raw)
#     reflection = questions[-1] if questions else raw

#     return {
#         "reflection": reflection
#     }



@app.post("/reflect")
async def reflect(req: ReflectRequest):
    text = req.text
    
    prompt = f"""<|im_start|>system
You are a calm journaling companion.<|im_end|>
<|im_start|>user
Journal: "{text}"
Ask one reflective question.<|im_end|>
<|im_start|>assistant
"""

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            LLAMA_SERVER_URL,
            json={
                "prompt": prompt,
                "n_predict": 64,
                "temperature": 0.7,
                "stop": ["<|im_end|>", "\n\n"],
                "stream": False
            }
        )
        
        data = response.json()
        raw = data.get("content", "").strip()
        
        # Extract question
        questions = re.findall(r'([A-Z][^?.!]*\?)', raw)
        reflection = questions[-1] if questions else raw
        
        return {
            "reflection": reflection,
            "raw": raw
        }





@app.post("/speak-local")
async def speak_local(req: SpeakRequest):
    """Generate speech and play it locally on the server"""
    
    audio_path = f"audio/temp_{uuid.uuid4()}.wav"
    model_path = PIPER_MODELS["lessac"]
    
    try:
        # Generate audio
        subprocess.run(
            [
                "piper",
                "--model", model_path,
                "--output_file", audio_path
            ],
            input=req.text.encode(),
            timeout=10,
            check=True
        )
        
        # Play it locally using afplay (macOS)
        subprocess.Popen(["afplay", audio_path])
        
        return {
            "status": "playing",
            "text": req.text
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")
