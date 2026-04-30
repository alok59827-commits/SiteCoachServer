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
        
        # ---> सेफ कटऑफ: 250 शब्द (ताकि आउटपुट के लिए ज़्यादा टोकन बच सकें) <---
        words = transcript.split()
        if len(words) > 250:
            transcript = " ".join(words[:250]) + "..."
            
        # ---> एंटी-रिपीट प्रॉम्प्ट: AI को रिपीट करने से रोकने के लिए <---
        system_prompt = """You are a strict and professional communication coach for Site Engineers. 
Analyze the provided Hindi transcript ONLY for communication skills (professionalism, tone, clarity, and respect).
DO NOT assume or hallucinate background context.
Provide a score (0-100).

CRITICAL RULES:
1. Provide a MAXIMUM of 3 to 4 points for mistakes, improvements, and action_items.
2. DO NOT REPEAT any point. Every bullet point MUST be unique.
3. Group similar issues into a single point.
4. Output MUST be strictly in JSON format with keys: score, mistakes, improvements, action_items. 
5. All text values MUST be in natural Hindi."""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.1, # इसे 0.1 किया ताकि AI एकदम सटीक और टू-द-पॉइंट रहे
            max_tokens=3000 # आउटपुट के लिए भरपूर जगह दी
        )
        
        result_json = json.loads(chat_completion.choices[0].message.content)
        
        return {
            "success": True,
            "transcript": transcript,
            "coaching_feedback": result_json
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
