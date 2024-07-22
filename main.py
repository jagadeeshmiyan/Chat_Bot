from fastapi import FastAPI, File, UploadFile, HTTPException
import fitz  # PyMuPDF
from typing import List
import openai
import os

app = FastAPI()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(file: UploadFile):
    try:
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def get_llm_response(query: str, context: str):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    text = extract_text_from_pdf(file)
    return {"text": text}

@app.post("/chat_with_pdf/")
async def chat_with_pdf(query: str, pdf_text: str):
    response = get_llm_response(query, pdf_text)
    return {"response": response}

# Unit tests
from fastapi.testclient import TestClient

client = TestClient(app)

def test_upload_pdf():
    with open("sample.pdf", "rb") as f:
        response = client.post("/upload_pdf/", files={"file": ("sample.pdf", f, "application/pdf")})
    assert response.status_code == 200
    assert "text" in response.json()

def test_chat_with_pdf():
    response = client.post("/chat_with_pdf/", json={"query": "What is the summary?", "pdf_text": "This is a test PDF content."})
    assert response.status_code == 200
    assert "response" in response.json()
