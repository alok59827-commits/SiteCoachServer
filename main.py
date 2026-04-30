from fastapi import FastAPI, File, UploadFile
from deepgram import DeepgramClient, PrerecordedOptions
from groq import Groq
import os
import json

app = FastAPI()

# API Keys 
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

deepgram = DeepgramClient(DEEPGRAM_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
def read_root():
    return {"status": "Site Coach Server is Running Like a Rocket! 🚀"}

@app.post("/upload-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # 1. ऑडियो फाइल पढ़ें
        audio_data = await file.read()
        
        # 2. Deepgram से टेक्स्ट निकालें
        payload = {"buffer": audio_data}
        options = PrerecordedOptions(
            model="nova-2",
            language="hi",
            smart_format=True,
            diarize=True
        )
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = response.results.channels[0].alternatives[0].transcript
        
        # ---> सेफ्टी ब्रेक (लिमिट पार होने से रोकने के लिए) <---
        transcript = transcript[:12000] 
        
        # 3. Llama 3.1 को टेक्स्ट भेजें
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a communication coach for construction site engineers. The user will provide a transcribed Hindi audio text. Provide a score (0-100), mistakes, improvements, and action items. You MUST write your ENTIRE response strictly in natural Hindi (Devanagari script). Output MUST be in JSON format with keys: score, mistakes, improvements, action_items."
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ],
            model="llama-3.1-8b-instant", # नया और फास्ट मॉडल
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2000 # यह हमारी टोकन लिमिट है
        )
        
        result_json = json.loads(chat_completion.choices[0].message.content)
        
        return {
            "success": True,
            "transcript": transcript,
            "coaching_feedback": result_json
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
