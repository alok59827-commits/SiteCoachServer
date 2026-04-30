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
    return {"status": "Site Coach Server is Running! 🚀"}

@app.post("/upload-audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        audio_data = await file.read()
        
        payload = {"buffer": audio_data}
        options = PrerecordedOptions(
            model="nova-2",
            language="hi",
            smart_format=True,
            diarize=True # स्पीकर अलग करने का जादू
        )
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        
        alt = response.results.channels[0].alternatives[0]
        transcript = alt.transcript
        
        # 1. WhatsApp स्टाइल चैट के लिए डेटा तैयार करना (S1, S2 और Time)
        chat_dialogue = []
        if hasattr(alt, 'words') and alt.words:
            current_speaker = alt.words[0].speaker
            current_sentence = []
            start_time = alt.words[0].start
            
            for word in alt.words:
                if word.speaker == current_speaker:
                    current_sentence.append(word.punctuated_word)
                else:
                    mins, secs = int(start_time // 60), int(start_time % 60)
                    chat_dialogue.append({
                        "speaker": f"S{current_speaker + 1}",
                        "text": " ".join(current_sentence),
                        "time": f"{mins:02d}:{secs:02d}"
                    })
                    current_speaker = word.speaker
                    current_sentence = [word.punctuated_word]
                    start_time = word.start
                    
            # आख़िरी लाइन जोड़ना
            mins, secs = int(start_time // 60), int(start_time % 60)
            chat_dialogue.append({
                "speaker": f"S{current_speaker + 1}",
                "text": " ".join(current_sentence),
                "time": f"{mins:02d}:{secs:02d}"
            })
        else:
            chat_dialogue.append({"speaker": "S1", "text": transcript, "time": "00:00"})

        # 2. Llama 3 के लिए लिमिट (ताकि क्रैश न हो)
        words_list = transcript.split()
        llama_transcript = " ".join(words_list[:250]) + "..." if len(words_list) > 250 else transcript

        # 3. AI प्रॉम्प्ट (अब Summary और Learn Points भी मांगेगा)
        system_prompt = """You are a professional communication coach. 
Analyze the Hindi transcript for professionalism, tone, and clarity.
Provide a score (0-100).
CRITICAL RULES:
1. Provide max 3-4 points for mistakes, improvements, and action_items.
2. Provide a 'summary' (2-3 sentences explaining the core discussion).
3. Provide 'learn_points' (2 practical tips for engineers to learn from this).
4. Output MUST be strictly JSON with keys: score, mistakes, improvements, action_items, summary, learn_points.
5. All text MUST be in natural Hindi."""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": llama_transcript}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=3000
        )
        
        result_json = json.loads(chat_completion.choices[0].message.content)
        
        return {
            "success": True,
            "transcript": transcript,
            "chat_dialogue": chat_dialogue, # UI के लिए नया हथियार
            "coaching_feedback": result_json
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
