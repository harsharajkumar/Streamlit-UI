import streamlit as st
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from PyPDF2 import PdfReader
import re
import nltk
import time
import gc

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Constants
model_name = "ProphetNet"
summary_path = "summarization"
story_path = "story_generation"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Functions (same as yours, copy-paste them here)
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    return text.strip()

def chunk_text(text, max_tokens=1000):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text(text, tokenizer, model):
    n = len(text.split()) // 10
    prompt = f"Summarize this part of the research paper to less than {n} words:\n{text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(**inputs, max_new_tokens=n+100, num_return_sequences=1)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_story(summary, tokenizer, model):
    story_prompt = f"""You are a master storyteller...
--- Summary: {summary} ..."""  # use full prompt as in your script
    max_tokens = min(1024, tokenizer.model_max_length - 20)
    inputs = tokenizer(story_prompt, return_tensors="pt", truncation=False, max_length=max_tokens).to(device)
    story_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_return_sequences=1,
        temperature=0.7,
        repetition_penalty=1.1
    )
    return tokenizer.decode(story_ids[0], skip_special_tokens=True)

# Streamlit UI
st.set_page_config(page_title="Research Story Generator", layout="centered")
st.title("📚 Research Paper to Story Generator")

uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and cleaning text..."):
        text = extract_text_from_pdf(uploaded_file)
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned)

    with st.spinner("Loading summarization model..."):
        summary_tokenizer = AutoTokenizer.from_pretrained(os.path.join(summary_path, model_name))
        summary_model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(summary_path, model_name)).to(device)

    with st.spinner("Summarizing chunks..."):
        summaries = [summarize_text(chunk, summary_tokenizer, summary_model) for chunk in chunks]
        combined_summary = " ".join(summaries)
        del summary_model, summary_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    with st.spinner("Loading story generation model..."):
        story_tokenizer = AutoTokenizer.from_pretrained(os.path.join(story_path, model_name))
        story_model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(story_path, model_name)).to(device)

    with st.spinner("Generating story..."):
        story = generate_story(combined_summary, story_tokenizer, story_model)
        del story_model, story_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    st.success("✅ Story Generated!")
    st.subheader("📝 Generated Story:")
    st.markdown(story)
