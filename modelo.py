from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import torch

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

# Carregar modelo de embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Carregar modelo de linguagem (Gemma Instruct)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    torch_dtype=torch.bfloat16
)

# Pipeline de geração
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device=-1  # ou -1 para CPU
)

# Gerar embeddings dos documentos
doc_embeddings = embedder.encode(docs, normalize_embeddings=True)

# FastAPI app
app = FastAPI(title="Chatbot Sidik")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["http://127.0.0.1:5500"] se usares um servidor local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Question):
    question = payload.question

    # Embedding da pergunta
    q_embedding = embedder.encode([question], normalize_embeddings=True)

    # Similaridade com os documentos
    scores = cosine_similarity(q_embedding, doc_embeddings)[0]

    # Verificar se existe contexto relevante
    max_score = scores.max()
    threshold = 0.3  # Ajuste conforme necessidade

    if max_score < threshold:
        return {
            "question": question,
            "context": "",
            "answer": "Desculpe, não tenho informação suficiente para responder a essa pergunta."
        }

    # Selecionar os 4 documentos mais relevantes
    top_k = 4
    top_indices = scores.argsort()[-top_k:][::-1]
    context = "\n".join([docs[i] for i in top_indices])

    # Montar prompt
    prompt = f"""
Com base nas informações abaixo, responda à pergunta em português, de forma direta, clara e em primeira pessoa.

Informações:
{context}

Pergunta: {question}
Resposta:
""".strip()

    # Gerar resposta com o modelo, corrigindo warnings
    output = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        top_p=1.0,
        top_k=0
    )[0]["generated_text"]

    response = output.replace(prompt, "").strip()

    return {"question": question, "context": context, "answer": response}
