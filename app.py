
import gradio as gr
import whisper
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
from huggingface_hub import login

# Login with HF token stored as Space secret
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✅ Logged in to HuggingFace")

# ── Load models on startup ────────────────────────────────────────────────────
print("Loading Whisper medium...")
whisper_model = whisper.load_model("medium")
print("✅ Whisper loaded")

print("Loading fine-tuned Gemma 3 1B...")
BASE_MODEL = "unsloth/gemma-3-1b-it"  # ungated version — no auth needed
ADAPTER    = "charan-reddy222/medscribe-rural-gemma3-1b"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER, token=HF_TOKEN)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype = torch.float32,
    device_map  = "auto",
    token        = HF_TOKEN,
)
model = PeftModel.from_pretrained(base_model, ADAPTER, token=HF_TOKEN)
model.eval()
print("✅ Gemma loaded")

# ── Helper functions ──────────────────────────────────────────────────────────
def format_note(note_text):
    try:
        if "```json" in note_text:
            note_text = note_text.split("```json")[1].split("```")[0].strip()
        note = json.loads(note_text)
        html = "<div style='font-family: Arial; padding: 10px;'>"
        html += "<h2 style='color: #2c7a4b;'>🏥 MedScribe Rural — Clinical Note</h2><hr>"
        for section, content in note.items():
            html += f"<h3 style='color: #1a5276;'>{section.replace('_', ' ').upper()}</h3>"
            if isinstance(content, dict):
                for k, v in content.items():
                    if isinstance(v, list):
                        html += f"<b>{k}:</b><ul>" + "".join(f"<li>{i}</li>" for i in v) + "</ul>"
                    else:
                        html += f"<b>{k}:</b> {v}<br>"
            elif isinstance(content, list):
                html += "<ul>" + "".join(f"<li>{i}</li>" for i in content) + "</ul>"
            else:
                html += f"<p>{content}</p>"
        html += "</div>"
        return html
    except:
        return f"<pre style='padding:10px'>{note_text}</pre>"

def process_audio(audio_path):
    if audio_path is None:
        return "⚠️ Please record or upload audio first.", ""
    try:
        result = whisper_model.transcribe(
            audio_path, language="te", task="transcribe",
            fp16=torch.cuda.is_available()
        )
        transcript = result["text"].strip()

        instruction = (
            "You are MedScribe Rural, an AI clinical assistant for rural Indian clinics. "
            "You will receive a Telugu-English doctor-patient conversation transcript. "
            "Generate a structured clinical note in JSON format with sections: "
            "patient_info, chief_complaint, history_of_present_illness, "
            "past_medical_history, allergies, physical_examination, assessment, plan."
        )
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{transcript}

### Response:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024,
                temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        note = full.split("### Response:")[-1].strip() if "### Response:" in full else full
        return transcript, format_note(note)
    except Exception as e:
        return f"Error: {str(e)}", ""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="MedScribe Rural") as demo:
    gr.HTML("""
        <div style='text-align:center; padding:20px; background:#e8f5e9; border-radius:10px;'>
            <h1>🏥 MedScribe Rural</h1>
            <p>AI clinical note generation for rural Indian clinics</p>
            <p><b>Speak in Telugu + English</b> → Structured clinical notes instantly</p>
            <p style='color:gray; font-size:12px'>Gemma 3 1B (fine-tuned) + Whisper Medium</p>
        </div>
    """)
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎙️ Record or Upload Audio")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Doctor-Patient Conversation (Telugu + English)",
            )
            gr.Markdown("""
            **Try saying:**
            - *నమస్కారం డాక్టర్ గారు, నాకు జ్వరం వస్తుంది*
            - *Rendu rojulu nundi fever undi, daggulu vastundi*
            """)
            submit_btn = gr.Button("🔬 Generate Clinical Note", variant="primary", size="lg")
        with gr.Column():
            gr.Markdown("### 📝 Transcript")
            transcript_out = gr.Textbox(label="Transcribed Text", lines=8)
    gr.Markdown("### 📋 Clinical Note")
    note_out = gr.HTML("<p style='color:gray'>Clinical note will appear here...</p>")
    gr.HTML("""
        <div style='text-align:center; color:gray; font-size:12px; margin-top:20px;'>
            MedScribe Rural — Gemma 4 Good Hackathon 2026 | Built for rural healthcare in India<br>
            ⚠️ For demonstration only. Not a substitute for professional medical judgment.
        </div>
    """)
    submit_btn.click(fn=process_audio, inputs=[audio_input], outputs=[transcript_out, note_out])

demo.launch()
