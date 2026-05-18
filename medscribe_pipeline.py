
import whisper
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# ── Load models ───────────────────────────────────────────────────────────────
def load_models(base_model_name="unsloth/gemma-3-1b-it",
                adapter_path="charan-reddy222/medscribe-rural-gemma3-1b"):
    """Load Whisper and fine-tuned Gemma models."""
    
    print("Loading Whisper...")
    whisper_model = whisper.load_model("medium")
    
    print("Loading Gemma + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return whisper_model, model, tokenizer

# ── Transcribe audio ──────────────────────────────────────────────────────────
def transcribe_audio(whisper_model, audio_path):
    """Transcribe Telugu+English audio to text."""
    result = whisper_model.transcribe(
        audio_path,
        language="te",
        task="transcribe",
        fp16=torch.cuda.is_available(),
    )
    return result["text"].strip()

# ── Generate clinical note ────────────────────────────────────────────────────
def generate_note(model, tokenizer, transcript):
    """Generate structured clinical note from transcript."""
    
    instruction = (
        "You are MedScribe Rural, an AI clinical assistant for rural Indian clinics. "
        "You will receive a Telugu-English (code-switched) doctor-patient conversation transcript. "
        "Generate a structured clinical note in JSON format with the following sections: "
        "patient_info, chief_complaint, history_of_present_illness, review_of_systems, "
        "past_medical_history, medications_on_arrival, allergies, physical_examination, "
        "assessment, plan (medications, follow_up, patient_education). "
        "Be concise, clinically accurate, and use standard medical abbreviations."
    )
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
    prompt = alpaca_prompt.format(instruction, transcript)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full_output:
        return full_output.split("### Response:")[-1].strip()
    return full_output

# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(whisper_model, model, tokenizer, audio_path):
    """End-to-end: audio file → clinical note."""
    transcript = transcribe_audio(whisper_model, audio_path)
    note = generate_note(model, tokenizer, transcript)
    return transcript, note
