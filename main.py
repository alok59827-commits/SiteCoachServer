from fastapi import FastAPI, File, UploadFile
from deepgram import DeepgramClient, PrerecordedOptions
from groq import Groq
import os
import json

app = FastAPI()

# API Keys (हम इन्हें बाद में वेबसाइट पर सुरक्षित तरीके से डालेंगे, यहाँ नहीं)
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Clients (औज़ार) चालू करना
deepgram = DeepgramClient(DEEPGRAM_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# यह चेक करने के लिए कि सर्वर चल रहा है या नहीं
@app.get("/")
def read_root():
    return {"status": "Site Coach Server is Running Like a Rocket! 🚀"}

# ऑडियो अपलोड वाला मेन फंक्शन
@app.post("/upload-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        # 1. ऐप से आई हुई ऑडियो फाइल को पढ़ें
        audio_data = await file.read()
        
        # 2. Deepgram को फाइल भेजें (सफाई और एकदम सटीक टेक्स्ट के लिए)
        payload = {"buffer": audio_data}
        options = PrerecordedOptions(
            model="nova-2",
            language="hi",
            smart_format=True,
            diarize=True # यह अलग-अलग आवाज़ों को पहचानेगा
        )
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        transcript = response.results.channels[0].alternatives[0].transcript
        
        # 3. Groq (Llama 3) को टेक्स्ट भेजें (कोचिंग के लिए)
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
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=4096
        )
        
        result_json = json.loads(chat_completion.choices[0].message.content)
        
        # 4. फाइनल जवाब वापस Android ऐप को दें
        return {
            "success": True,
            "transcript": transcript,
            "coaching_feedback": result_json
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
