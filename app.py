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

# === Functions ===
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

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
    story_prompt = f"""You are a master storyteller. Convert the following technical summary into a story that captures imagination, emotion, and human connection.

--- Summary: {summary}"""
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

# === UI Starts Here ===
st.set_page_config(page_title="🧠 Paper2Story - Research Paper to Story", layout="centered")
st.markdown(
    """
    <style>
        .main {background-color: #f5f7fa;}
        .block-container {padding-top: 2rem;}
        .stSpinner > div > div {color: #4B8BBE;}
        .stButton>button {background-color: #4B8BBE; color: white; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=80)
st.sidebar.title("🛠️ How it works")
st.sidebar.markdown(
    """
    1. Upload a research paper (PDF)  
    2. We'll extract and clean the content  
    3. Summarize it intelligently  
    4. Craft a human-friendly story 🌟  
    """
)
st.sidebar.markdown("Made with ❤️ for Capstone")

# Title
st.title("📚 Research Paper ➜ Story Generator")
st.caption("Transform complex academic writing into engaging narratives using AI.")

# Upload
uploaded_file = st.file_uploader("📤 Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Extracting & cleaning text..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)

    with st.spinner("⚙️ Loading summarization model..."):
        summary_tokenizer = AutoTokenizer.from_pretrained(os.path.join(summary_path, model_name))
        summary_model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(summary_path, model_name)).to(device)

    summaries = []
    with st.spinner("🧠 Summarizing paper..."):
        progress = st.progress(0)
        for idx, chunk in enumerate(chunks):
            summaries.append(summarize_text(chunk, summary_tokenizer, summary_model))
            progress.progress((idx + 1) / len(chunks))
        combined_summary = " ".join(summaries)

    del summary_model, summary_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    with st.spinner("📖 Loading story generation model..."):
        story_tokenizer = AutoTokenizer.from_pretrained(os.path.join(story_path, model_name))
        story_model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(story_path, model_name)).to(device)

    with st.spinner("🪄 Generating your story..."):
        story = generate_story(combined_summary, story_tokenizer, story_model)

    del story_model, story_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    st.success("✅ Story Generated!")
    st.subheader("✨ Your Generated Story")
    with st.expander("Click to read the full story"):
        st.markdown(story)

    st.download_button("💾 Download Story", story, file_name="generated_story.txt")

else:
    st.info("👆 Please upload a PDF to begin.")

