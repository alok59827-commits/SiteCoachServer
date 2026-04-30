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
        audio_data = await file.read()
        
        payload = {"buffer": audio_data}
        options = PrerecordedOptions(
            model="nova-2",
            language="hi",
            smart_format=True,
            diarize=True
        )
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = response.results.channels[0].alternatives[0].transcript
        
        # ---> स्मार्ट कटऑफ: अब हम कैरेक्टर नहीं, 350 शब्दों (Words) पर रोक लगाएंगे <---
        words = transcript.split()
        if len(words) > 350:
            transcript = " ".join(words[:350]) + "..."
            
        # ---> स्ट्रिक्ट प्रॉम्प्ट: AI को सख्त हिदायत <---
        system_prompt = """You are a strict and professional communication coach for Site Engineers. 
Analyze the provided Hindi transcript ONLY for communication skills (professionalism, tone, clarity, grammar, and respect).
DO NOT assume or hallucinate background context (e.g., weather, site conditions, personal details).
Provide a score (0-100).
Identify strictly professional mistakes (like using slang 'यार', being rude, or unclear sentences).
Provide practical actionable improvements.
Output MUST be strictly in JSON format with keys: score, mistakes, improvements, action_items. All values MUST be in Hindi."""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.2, # इसे 0.2 किया ताकि AI सटीक रहे पर भाषा अच्छी लिखे
            max_tokens=1500
        )
        
        result_json = json.loads(chat_completion.choices[0].message.content)
        
        return {
            "success": True,
            "transcript": transcript,
            "coaching_feedback": result_json
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
