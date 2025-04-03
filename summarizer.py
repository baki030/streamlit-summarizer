import streamlit as st
from transformers import pipeline
import pdfplumber
import requests
from bs4 import BeautifulSoup

# Load summarization model
summarizer = pipeline("summarization", model="t5-small")

def extract_text_from_url(url):
    """Extract text from an article using BeautifulSoup"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return " ".join([p.get_text() for p in paragraphs])

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file"""
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def summarize_text(text, max_length=150):
    """Generate a summary with a specified length"""
    if len(text) < 50:
        return "Text too short for summarization."
    return summarizer(text, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']

# Streamlit UI
st.title("Custom Content Summarizer")
st.write("Summarize articles, PDFs, or custom text with AI.")

# Input options
source = st.radio("Choose input type:", ("Text", "URL", "PDF Upload"))

if source == "Text":
    user_text = st.text_area("Enter your text:")
    length = st.slider("Summary Length", 50, 300, 150)
    if st.button("Summarize"):
        st.write(summarize_text(user_text, max_length=length))

elif source == "URL":
    url = st.text_input("Enter article URL:")
    length = st.slider("Summary Length", 50, 300, 150)
    if st.button("Summarize"):
        extracted_text = extract_text_from_url(url)
        st.write(summarize_text(extracted_text, max_length=length))

elif source == "PDF Upload":
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    length = st.slider("Summary Length", 50, 300, 150)
    if uploaded_file and st.button("Summarize"):
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.write(summarize_text(extracted_text, max_length=length))


### **4. Deploy for Free**
- Use **Streamlit Cloud**: Upload your script to GitHub and deploy it via [Streamlit Community Cloud](https://streamlit.io/cloud).
- Use **Google Colab**: Run it as a notebook for free.
- Use **Hugging Face Spaces**: Deploy it easily using [Gradio](https://huggingface.co/spaces).

This setup allows you to summarize articles, research papers, and documents **without paying for APIs**. Want more customization, like keyword-based summaries? You can add **spaCy** or **NLTK** for better text processing.
