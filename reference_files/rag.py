import os
from typing import List

# LangChain
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever

# ========= CONFIG =========
PDF_PATH = "regimento_interno_2024.pdf"  # coloque o caminho do seu PDF
# Modelo de LLM no Ollama (ajuste conforme seu ambiente)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Embeddings multilíngue (bom p/ pt-BR)
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# Recuperação
TOP_K = 4
USE_MMR = True  # MMR ajuda a diversificar
# ==========================


def load_and_chunk(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # cada página vira um Document (metadata['page'] começa em 0)

    # prefixa página em metadata numa chave amigável
    for d in docs:
        if "page" in d.metadata:
            d.metadata["page_number"] = d.metadata["page"] + 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL, encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.from_documents(chunks, embeddings)
    return vs


def make_retriever(vs):
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    if USE_MMR:
        retriever = vs.as_retriever(
            # search_type define a estratégia de busca:
            # "mmr" = Maximal Marginal Relevance → traz resultados relevantes e ao mesmo tempo diversos
            search_type="mmr",

            search_kwargs={
                # k = número final de chunks que você quer recuperar
                "k": TOP_K,

                # fetch_k = número inicial de candidatos que o FAISS vai buscar antes de aplicar o MMR
                # (quanto maior, mais diversidade, mas também mais custo de cálculo)
                "fetch_k": max(8, TOP_K * 2)
            }
        )

        # ----- RERANKEADOR -----
        # Opção A: passar só o nome do modelo
        # reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=TOP_K)

        # Opção B: passar a instância do HuggingFaceCrossEncoder via 'model'
        ce = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
        # Cria um reranqueador baseado em CrossEncoder
        reranker = CrossEncoderReranker(
            # Aqui usamos a instância `ce` já carregada.
            model=ce,
            # top_n → número de documentos (chunks) que você quer manter após o reranqueamento.
            # Exemplo: se o retriever trouxe 10 candidatos, o CrossEncoder os reordena
            # e devolve apenas os TOP_K (mais relevantes).
            top_n=TOP_K
        )

        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever,
        )
    return retriever

def format_docs(docs: List) -> str:
    # Junta os docs, incluindo a fonte (página)
    parts = []
    for d in docs:
        page = d.metadata.get("page_number", d.metadata.get("page", ""))
        src = f"(p.{page})" if page else ""
        parts.append(f"{src} {d.page_content}".strip())
    return "\n\n---\n\n".join(parts)

def build_chain(retriever):
    # Prompt de RAG (instruções + contexto + pergunta)
    system_msg = (
        "Você é um assistente acadêmico. Responda de forma precisa, cite as páginas entre parênteses "
        "quando usar trechos do contexto. Se a resposta não estiver no contexto, diga que não encontrou."
    )
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human",
             "Pergunta:\n{question}\n\n"
             "Contexto (trechos do documento):\n{context}\n\n"
             "Responda de forma direta e cite as páginas relevantes.")
        ]
    )

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

    # Pipeline: { "docs": retriever(question), "question": passthrough } -> formatação -> LLM
    rag_chain = (
        RunnableParallel(
            {"docs": retriever, "question": RunnablePassthrough()}
        )
        | {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | template
        | llm
        | StrOutputParser()
    )
    return rag_chain


def main():
    print(">> Carregando e chunkando PDF...")
    chunks = load_and_chunk(PDF_PATH)

    print(f">> Total de chunks: {len(chunks)}")
    print(">> Construindo FAISS...")
    vs = build_vectorstore(chunks)

    print(">> Preparando retriever (MMR e reranker)...")
    retriever = make_retriever(vs)

    print(">> Montando cadeia RAG...")
    chain = build_chain(retriever)

    resposta = chain.invoke(input("Faça sua pergunta: "))
    print(resposta)

if __name__ == "__main__":
    main()