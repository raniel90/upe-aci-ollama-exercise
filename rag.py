import os
import sys
from typing import List, Optional
from pathlib import Path

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


class Config:
    """Configurações do sistema RAG"""
    
    BASE_DIR = Path(__file__).parent
    PDF_PATH = BASE_DIR / "data" / "nr-06-atualizada-2022-1.pdf"
    VECTORSTORE_PATH = BASE_DIR / "vectorstore"
    
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TEMPERATURE = 0.0
    
    EMBEDDING_MODEL = os.environ.get(
        "EMBEDDING_MODEL", 
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 200
    
    TOP_K = 4
    FETCH_K = 10
    USE_MMR = True
    USE_RERANKER = True
    
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


def verificar_pdf() -> bool:
    """Verifica existência do PDF"""
    if not Config.PDF_PATH.exists():
        print(f"Erro: PDF não encontrado em {Config.PDF_PATH}")
        return False
    print(f"PDF encontrado: {Config.PDF_PATH.name}")
    return True


def carregar_e_dividir_documento(pdf_path: Path) -> List:
    """Carrega PDF e divide em chunks"""
    print(f"\nCarregando: {pdf_path.name}")
    
    loader = PyPDFLoader(str(pdf_path))
    documentos = loader.load()
    
    print(f"Páginas carregadas: {len(documentos)}")
    
    for doc in documentos:
        if "page" in doc.metadata:
            doc.metadata["page_number"] = doc.metadata["page"] + 1
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_documents(documentos)
    print(f"Chunks criados: {len(chunks)}")
    
    return chunks


def construir_vectorstore(chunks: List, caminho_salvar: Optional[Path] = None):
    """Constrói vectorstore FAISS"""
    print(f"\nCriando embeddings: {Config.EMBEDDING_MODEL.split('/')[-1]}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("Construindo vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    if caminho_salvar:
        caminho_salvar.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(caminho_salvar))
        print(f"Vectorstore salvo: {caminho_salvar.name}")
    
    return vectorstore


def carregar_vectorstore_existente(caminho: Path):
    """Carrega vectorstore existente"""
    if not caminho.exists():
        return None
    
    print(f"\nCarregando vectorstore: {caminho.name}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectorstore = FAISS.load_local(
        str(caminho), 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore


def criar_retriever(vectorstore):
    """Configura retriever com MMR e reranking"""
    print(f"\nConfigurando retriever (MMR: {Config.USE_MMR}, Reranker: {Config.USE_RERANKER})")
    
    if Config.USE_MMR:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": Config.TOP_K,
                "fetch_k": Config.FETCH_K,
            }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": Config.TOP_K}
        )
    
    if Config.USE_RERANKER:
        cross_encoder = HuggingFaceCrossEncoder(
            model_name=Config.RERANKER_MODEL
        )
        
        reranker = CrossEncoderReranker(
            model=cross_encoder,
            top_n=Config.TOP_K
        )
        
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever
        )
    
    return retriever


def formatar_documentos(docs: List) -> str:
    """Formata documentos recuperados"""
    partes = []
    for doc in docs:
        page = doc.metadata.get("page_number", doc.metadata.get("page", ""))
        fonte = f"[Página {page}]" if page else "[Fonte desconhecida]"
        partes.append(f"{fonte}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(partes)


def construir_cadeia_rag(retriever):
    """Constrói cadeia RAG"""
    print(f"\nModelo LLM: {Config.OLLAMA_MODEL}")
    
    system_prompt = """Você é um assistente especializado na Norma Regulamentadora NR-06 
(Equipamentos de Proteção Individual - EPI) do Brasil.

INSTRUÇÕES:
1. Responda perguntas com base APENAS no contexto fornecido
2. Organize a resposta de forma clara e estruturada
3. Cite sempre as páginas entre colchetes: [Página X]
4. Use linguagem técnica mas acessível
5. Se a informação não estiver no contexto, diga: "Esta informação não está disponível nos trechos consultados."

Você está auxiliando profissionais de segurança do trabalho."""

    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """Com base no contexto abaixo, responda à pergunta.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:""")
    ])
    
    llm = ChatOllama(
        model=Config.OLLAMA_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        temperature=Config.OLLAMA_TEMPERATURE
    )
    
    rag_chain = (
        RunnableParallel({
            "docs": retriever,
            "question": RunnablePassthrough()
        })
        | {
            "context": lambda x: formatar_documentos(x["docs"]),
            "question": lambda x: x["question"]
        }
        | template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


class AgenteNR06:
    """Agente RAG para NR-06"""
    
    def __init__(self, recriar_vectorstore: bool = False):
        self.chain = None
        self.vectorstore = None
        
        if not verificar_pdf():
            raise FileNotFoundError(f"PDF não encontrado: {Config.PDF_PATH}")
        
        if not recriar_vectorstore:
            self.vectorstore = carregar_vectorstore_existente(Config.VECTORSTORE_PATH)
        
        if self.vectorstore is None:
            chunks = carregar_e_dividir_documento(Config.PDF_PATH)
            self.vectorstore = construir_vectorstore(chunks, Config.VECTORSTORE_PATH)
        
        retriever = criar_retriever(self.vectorstore)
        self.chain = construir_cadeia_rag(retriever)
    
    def perguntar(self, pergunta: str) -> str:
        """Processa pergunta e retorna resposta"""
        if not self.chain:
            raise RuntimeError("Cadeia RAG não inicializada")
        
        return self.chain.invoke(pergunta)
    
    def modo_interativo(self):
        """Interface interativa via terminal"""
        print("\n" + "="*70)
        print("AGENTE NR-06")
        print("="*70)
        print("\nComandos: 'sair' para encerrar, 'limpar' para limpar tela")
        print("-"*70 + "\n")
        
        while True:
            try:
                pergunta = input("Pergunta: ").strip()
                
                if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
                    print("\nEncerrando...")
                    break
                
                if pergunta.lower() in ['limpar', 'clear', 'cls']:
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                if not pergunta:
                    continue
                
                print("\nProcessando...\n")
                resposta = self.perguntar(pergunta)
                
                print(f"Resposta:\n{resposta}\n")
                print("-"*70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrompido.")
                break
            except Exception as e:
                print(f"\nErro: {e}\n")
                continue


def main():
    """Função principal"""
    print("\n" + "="*70)
    print("Iniciando RAG Agent - NR-06")
    print("="*70)
    
    try:
        recriar = "--recriar" in sys.argv or "--rebuild" in sys.argv
        
        if recriar:
            print("\nModo: Recriando vectorstore")
        
        agente = AgenteNR06(recriar_vectorstore=recriar)
        agente.modo_interativo()
        
    except KeyboardInterrupt:
        print("\n\nInterrompido.")
        sys.exit(0)
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()