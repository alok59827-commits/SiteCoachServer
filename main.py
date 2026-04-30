from fastapi import FastAPI, File, UploadFile, Form
from deepgram import DeepgramClient, PrerecordedOptions
from groq import Groq
import os
import json

app = FastAPI()

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

deepgram = DeepgramClient(DEEPGRAM_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

@app.get("/")
def read_root():
    return {"status": "Site Coach Server is Running with Fixed JSON! 🌍🚀"}

@app.post("/upload-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    audience: str = Form("Peer/Colleague"),
    output_language: str = Form("Hindi")
):
    try:
        audio_data = await file.read()
        
        payload = {"buffer": audio_data}
        options = PrerecordedOptions(
            model="nova-2",
            detect_language=True,
            smart_format=True,
            diarize=True,
            keywords=["DBL", "जल निगम", "Rising main", "Shuttering", "JCB", "Contractor", "Panchayat", "Plinth", "PCC"]
        )
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        
        alt = response.results.channels[0].alternatives[0]
        transcript = alt.transcript
        
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
                    
            mins, secs = int(start_time // 60), int(start_time % 60)
            chat_dialogue.append({
                "speaker": f"S{current_speaker + 1}",
                "text": " ".join(current_sentence),
                "time": f"{mins:02d}:{secs:02d}"
            })
        else:
            chat_dialogue.append({"speaker": "S1", "text": transcript, "time": "00:00"})

        words_list = transcript.split()
        llama_transcript = " ".join(words_list[:250]) + "..." if len(words_list) > 250 else transcript

        # 🚨 यहाँ हमने बीमारी का पक्का इलाज कर दिया है
        system_prompt = f"""You are a strict communication coach for Site Engineers.
The engineer in the audio is talking to their: '{audience}'.

CRITICAL RULES:
1. LANGUAGE: You MUST provide all text STRICTLY in: {output_language}.
2. SCORE FORMAT: The 'score' MUST be a single integer number between 0 and 100 (e.g., 65). DO NOT write fractions like '6/10'.
3. JSON FORMAT: Your output MUST be a valid JSON object. Do not include trailing commas.

EXPECTED JSON STRUCTURE:
{{
  "score": 65,
  "mistakes": ["mistake 1", "mistake 2"],
  "improvements": ["improvement 1", "improvement 2"],
  "action_items": ["action 1"],
  "summary": "Short summary text here.",
  "learn_points": ["tip 1", "tip 2"]
}}"""

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
            "chat_dialogue": chat_dialogue,
            "coaching_feedback": result_json
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
