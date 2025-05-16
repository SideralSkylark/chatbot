from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# Dados pessoais
docs = [
    "Chamo-me Sidik Ebrahim Serage.",
    "Meu nome completo é Sidik Ebrahim Serage.",
    "Eu sou Sidik.",
    "Meu nome é Sidik Ebrahim Serage.",
    "Nasci em 16 de Janeiro de 2005 na Beira, Moçambique.",
    "Estudo na Universidade Católica de Moçambique.",
    "Estudo tecnologias de informacao",
    "Tenho interesse em Inteligência Artificial, Backend e Redes.",
    "Atualmente estou a trabalhar no WorkBridge, um marketplace de serviços.",
    "Uso tecnologias como Java, Spring Boot, Angular, Docker e JWT.",
    "Nos meus tempos livres gosto de jogar videojogos."
]

# Carregar modelos
embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Pré-processar os embeddings dos documentos
doc_embeddings = embedder.encode(docs, normalize_embeddings=True)

# FastAPI app
app = FastAPI(title="Chatbot Sidik")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Question):
    question = payload.question

    # Embedding da pergunta
    q_embedding = embedder.encode([question], normalize_embeddings=True)

    # Similaridade com todos os documentos
    scores = cosine_similarity(q_embedding, doc_embeddings)[0]

    # Pegar os 4 mais relevantes
    top_k = 4
    top_indices = scores.argsort()[-top_k:][::-1]
    context = "\n".join([docs[i] for i in top_indices])

    # Montar prompt melhorado
    prompt = f"""
    Com base nas informações abaixo, responda à pergunta em português, de forma direta, clara e em primeira pessoa.

    Informações:
    {context}

    Pergunta: {question}
    Resposta:
    """.strip()

    # Gerar resposta
    raw_output = generator(prompt, max_length=100, do_sample=False)[0]["generated_text"]
    response = raw_output.replace(prompt, "").strip()

    return {"question": question, "context": context, "answer": response}

# Rodar com: uvicorn modelo:app --reload
