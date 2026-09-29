import os
import sys
import sqlite3
import requests
import pandas as pd
from pypdf import PdfReader
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import chromadb
from openai import AzureOpenAI

# -------------------------------------------------------------
# 1. Validação de Credenciais do Azure OpenAI no Ambiente
# -------------------------------------------------------------
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

erros_config = []
if not AZURE_ENDPOINT:
    erros_config.append("AZURE_OPENAI_ENDPOINT não configurada.")
if not AZURE_API_KEY:
    erros_config.append("AZURE_OPENAI_API_KEY não configurada.")
if not AZURE_DEPLOYMENT:
    erros_config.append("AZURE_OPENAI_DEPLOYMENT_NAME não configurada.")

if erros_config:
    print(f"ERRO DE INICIALIZAÇÃO: {'; '.join(erros_config)}", file=sys.stderr)

azure_client = None
if not erros_config:
    azure_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION
    )

app = Flask(__name__)
DB_SQLITE = "historico_interacoes.db"

# -------------------------------------------------------------
# 2. SQLite3: Persistência de Perguntas e Respostas
# -------------------------------------------------------------
def init_sqlite():
    conn = sqlite3.connect(DB_SQLITE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pergunta TEXT NOT NULL,
            resposta TEXT,
            data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def salvar_interacao(pergunta, resposta):
    conn = sqlite3.connect(DB_SQLITE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO interacoes (pergunta, resposta) VALUES (?, ?)",
        (pergunta, resposta)
    )
    conn.commit()
    conn.close()

init_sqlite()

# -------------------------------------------------------------
# 3. Embeddings Locais & ChromaDB em Memória
# -------------------------------------------------------------
print("Carregando modelo de embeddings local...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="base_rag_educacional",
    metadata={"hnsw:space": "cosine"}
)

# -------------------------------------------------------------
# 4. Ingestão e Fatiamento de Conteúdo
# -------------------------------------------------------------
def fatiar_texto(texto, tamanho=500, overlap=50):
    chunks = []
    inicio = 0
    tamanho_total = len(texto)
    while inicio < tamanho_total:
        fim = min(inicio + tamanho, tamanho_total)
        chunk = texto[inicio:fim].strip()
        if chunk:
            chunks.append(chunk)
        inicio += (tamanho - overlap)
    return chunks

def extrair_web(url="https://gratuitos.netlify.app/"):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, "html.parser")
            for elemento in soup(["script", "style", "nav", "footer", "header"]):
                elemento.extract()
            linhas = (line.strip() for line in soup.get_text().splitlines())
            texto_limpo = " ".join(chunk for chunk in linhas if chunk)
            return texto_limpo
    except Exception as e:
        print(f"Aviso ao coletar web: {e}")
    return ""

def extrair_csv(caminho="dados.csv"):
    conteudo = []
    if os.path.exists(caminho):
        try:
            df = pd.read_csv(caminho)
            for _, row in df.iterrows():
                linha_formatada = " | ".join([f"{col}: {val}" for col, val in row.items()])
                conteudo.append(linha_formatada)
        except Exception as e:
            print(f"Aviso ao ler CSV: {e}")
    return "\n".join(conteudo)

def extrair_pdf(caminho="escola.pdf"):
    texto = []
    if os.path.exists(caminho):
        try:
            leitor = PdfReader(caminho)
            for pagina in leitor.pages:
                extraido = pagina.extract_text()
                if extraido:
                    texto.append(extraido)
        except Exception as e:
            print(f"Aviso ao ler PDF: {e}")
    return "\n".join(texto)

def indexar_documentos():
    todos_chunks = []
    metadados = []
    ids = []
    doc_id_counter = 0

    fontes = [
        ("Web Scraping (gratuitos.netlify.app)", extrair_web()),
        ("Planilha (dados.csv)", extrair_csv()),
        ("PDF Educacional (escola.pdf)", extrair_pdf())
    ]

    for origem, texto in fontes:
        if texto:
            chunks = fatiar_texto(texto, tamanho=500, overlap=50)
            for c in chunks:
                todos_chunks.append(c)
                metadados.append({"fonte": origem})
                ids.append(f"doc_{doc_id_counter}")
                doc_id_counter += 1

    if todos_chunks:
        embeddings = embed_model.encode(todos_chunks).tolist()
        collection.add(
            documents=todos_chunks,
            embeddings=embeddings,
            metadados=metadados,
            ids=ids
        )
        print(f"Total de {len(todos_chunks)} chunks indexados no ChromaDB com sucesso.")

indexar_documentos()

# -------------------------------------------------------------
# 5. Rotas do Flask
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/perguntar", methods=["POST"])
def perguntar():
    if erros_config:
        return jsonify({"erro": f"Serviço não configurado: {'; '.join(erros_config)}"}), 500

    dados = request.get_json() or {}
    pergunta_usuario = dados.get("pergunta", "").strip()

    if not pergunta_usuario:
        return jsonify({"erro": "A pergunta não pode estar vazia."}), 400

    # 1. Recuperação semântica (Retrieval)
    vetor_pergunta = embed_model.encode([pergunta_usuario]).tolist()
    resultados = collection.query(
        query_embeddings=vetor_pergunta,
        n_results=3
    )

    chunks_recuperados = []
    contexto_prompt = []

    if resultados and "documents" in resultados and resultados["documents"]:
        docs = resultados["documents"][0]
        metas = resultados["metadatas"][0] if "metadatas" in resultados else [{}] * len(docs)
        for doc, meta in zip(docs, metas):
            fonte = meta.get("fonte", "Desconhecida")
            chunks_recuperados.append({"conteudo": doc, "fonte": fonte})
            contexto_prompt.append(f"[Fonte: {fonte}]\n{doc}")

    texto_contexto = "\n\n".join(contexto_prompt)

    # 2. Prompt com restrição estrita de domínio
    prompt_sistema = (
        "Você é um tutor educacional estritamente factual. Responda à pergunta do usuário "
        "utilizando única e exclusivamente o contexto fornecido abaixo. Se a informação não estiver presente "
        "no contexto, diga claramente: 'Não encontrei dados suficientes no material consultado para responder a essa pergunta.' "
        "Não deduza, não alucine nem utilize conhecimentos prévios externos."
    )
    
    prompt_usuario_formatado = f"Contexto:\n{texto_contexto}\n\nPergunta: {pergunta_usuario}"

    # 3. Geração via Azure OpenAI
    try:
        resposta_llm = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario_formatado}
            ],
            temperature=0.0
        )
        resposta_final = resposta_llm.choices[0].message.content

        # 4. Auditoria e Persistência no SQLite3
        salvar_interacao(pergunta_usuario, resposta_final)

        return jsonify({
            "resposta": resposta_final,
            "auditoria": chunks_recuperados
        })

    except Exception as e:
        return jsonify({"erro": f"Falha na comunicação com o Azure OpenAI: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))