import os
import sys
from typing import List, Optional
from pathlib import Path

# LangChain imports
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


# ======================= CONFIGURAÇÕES =======================

class Config:
    """Configurações centralizadas do agente RAG"""
    
    # Caminhos
    BASE_DIR = Path(__file__).parent
    PDF_PATH = BASE_DIR / "data" / "nr-06-atualizada-2022-1.pdf"
    VECTORSTORE_PATH = BASE_DIR / "vectorstore"
    
    # Ollama
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TEMPERATURE = 0.0  # Temperatura zero para máxima precisão e consistência
    
    # Embeddings (modelo multilíngue otimizado para português)
    EMBEDDING_MODEL = os.environ.get(
        "EMBEDDING_MODEL", 
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Chunking - parâmetros otimizados para documentos regulatórios
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 200  # Overlap maior para preservar contexto
    
    # Retrieval
    TOP_K = 4  # Número de chunks a recuperar
    FETCH_K = 10  # Candidatos iniciais para MMR
    USE_MMR = True  # Usar Maximal Marginal Relevance
    USE_RERANKER = True  # Usar CrossEncoder para reranking
    
    # CrossEncoder
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


# ======================= FUNÇÕES PRINCIPAIS =======================

def verificar_pdf() -> bool:
    """Verifica se o PDF existe no caminho especificado"""
    if not Config.PDF_PATH.exists():
        print(f"❌ ERRO: PDF não encontrado em {Config.PDF_PATH}")
        print(f"   Certifique-se de que o arquivo está no local correto.")
        return False
    print(f"✓ PDF encontrado: {Config.PDF_PATH}")
    return True


def carregar_e_dividir_documento(pdf_path: Path) -> List:
    """
    Carrega o PDF e divide em chunks
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        
    Returns:
        Lista de documentos (chunks)
    """
    print(f"\n📄 Carregando PDF: {pdf_path.name}")
    
    # Carregar PDF
    loader = PyPDFLoader(str(pdf_path))
    documentos = loader.load()
    
    print(f"   └─ {len(documentos)} páginas carregadas")
    
    # Adicionar número de página em formato legível
    for doc in documentos:
        if "page" in doc.metadata:
            doc.metadata["page_number"] = doc.metadata["page"] + 1
    
    # Configurar text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )
    
    # Dividir em chunks
    chunks = splitter.split_documents(documentos)
    
    print(f"   └─ {len(chunks)} chunks criados (tamanho: {Config.CHUNK_SIZE}, overlap: {Config.CHUNK_OVERLAP})")
    
    return chunks


def construir_vectorstore(chunks: List, caminho_salvar: Optional[Path] = None):
    """
    Constrói o vectorstore FAISS com embeddings
    
    Args:
        chunks: Lista de documentos divididos
        caminho_salvar: Caminho para salvar o vectorstore (opcional)
        
    Returns:
        FAISS vectorstore
    """
    print(f"\n🔢 Criando embeddings com modelo: {Config.EMBEDDING_MODEL}")
    
    # Criar embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}  # Normalizar para melhor similaridade
    )
    
    # Construir vectorstore
    print("   └─ Construindo vectorstore FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Salvar se caminho especificado
    if caminho_salvar:
        caminho_salvar.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(caminho_salvar))
        print(f"   └─ Vectorstore salvo em: {caminho_salvar}")
    
    return vectorstore


def carregar_vectorstore_existente(caminho: Path):
    """
    Carrega um vectorstore FAISS já existente
    
    Args:
        caminho: Caminho do vectorstore salvo
        
    Returns:
        FAISS vectorstore ou None se não existir
    """
    if not caminho.exists():
        return None
    
    print(f"\n📂 Carregando vectorstore existente de: {caminho}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectorstore = FAISS.load_local(
        str(caminho), 
        embeddings,
        allow_dangerous_deserialization=True  # Necessário para FAISS
    )
    
    print("   └─ Vectorstore carregado com sucesso")
    return vectorstore


def criar_retriever(vectorstore):
    """
    Cria retriever com MMR e reranking
    
    Args:
        vectorstore: FAISS vectorstore
        
    Returns:
        Retriever configurado
    """
    print(f"\n🔍 Configurando retriever (MMR: {Config.USE_MMR}, Reranker: {Config.USE_RERANKER})")
    
    # Configurar retriever base
    if Config.USE_MMR:
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance
            search_kwargs={
                "k": Config.TOP_K,
                "fetch_k": Config.FETCH_K,
            }
        )
        print(f"   └─ MMR ativado (k={Config.TOP_K}, fetch_k={Config.FETCH_K})")
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": Config.TOP_K}
        )
        print(f"   └─ Busca por similaridade (k={Config.TOP_K})")
    
    # Adicionar reranker se habilitado
    if Config.USE_RERANKER:
        print(f"   └─ Adicionando reranker: {Config.RERANKER_MODEL}")
        
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
    """
    Formata documentos recuperados para o contexto
    
    Args:
        docs: Lista de documentos
        
    Returns:
        String formatada com contexto
    """
    partes = []
    for doc in docs:
        # Obter número da página
        page = doc.metadata.get("page_number", doc.metadata.get("page", ""))
        fonte = f"[Página {page}]" if page else "[Fonte desconhecida]"
        
        # Adicionar conteúdo formatado
        partes.append(f"{fonte}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(partes)


def construir_cadeia_rag(retriever):
    """
    Constrói a cadeia RAG completa
    
    Args:
        retriever: Retriever configurado
        
    Returns:
        Cadeia RAG pronta para uso
    """
    print(f"\n🔗 Construindo cadeia RAG com modelo: {Config.OLLAMA_MODEL}")
    
    # Prompt especializado para NR-06
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
    
    # Inicializar LLM
    llm = ChatOllama(
        model=Config.OLLAMA_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        temperature=Config.OLLAMA_TEMPERATURE
    )
    
    # Construir pipeline RAG
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
    
    print("   └─ Cadeia RAG construída com sucesso")
    
    return rag_chain


# ======================= CLASSE PRINCIPAL =======================

class AgenteNR06:
    """Agente RAG para interação com o documento NR-06"""
    
    def __init__(self, recriar_vectorstore: bool = False):
        """
        Inicializa o agente
        
        Args:
            recriar_vectorstore: Se True, recria o vectorstore mesmo se existir
        """
        self.chain = None
        self.vectorstore = None
        
        # Verificar PDF
        if not verificar_pdf():
            raise FileNotFoundError(f"PDF não encontrado: {Config.PDF_PATH}")
        
        # Carregar ou criar vectorstore
        if not recriar_vectorstore:
            self.vectorstore = carregar_vectorstore_existente(Config.VECTORSTORE_PATH)
        
        if self.vectorstore is None:
            # Criar novo vectorstore
            chunks = carregar_e_dividir_documento(Config.PDF_PATH)
            self.vectorstore = construir_vectorstore(chunks, Config.VECTORSTORE_PATH)
        
        # Criar retriever e cadeia
        retriever = criar_retriever(self.vectorstore)
        self.chain = construir_cadeia_rag(retriever)
    
    def perguntar(self, pergunta: str) -> str:
        """
        Faz uma pergunta ao agente
        
        Args:
            pergunta: Pergunta em linguagem natural
            
        Returns:
            Resposta do agente
        """
        if not self.chain:
            raise RuntimeError("Cadeia RAG não inicializada")
        
        return self.chain.invoke(pergunta)
    
    def modo_interativo(self):
        """Inicia modo de conversação interativa"""
        print("\n" + "="*70)
        print("🤖 AGENTE NR-06 - Assistente sobre Equipamentos de Proteção Individual")
        print("="*70)
        print("\nDigite suas perguntas sobre a NR-06.")
        print("Comandos especiais:")
        print("  • 'sair' ou 'exit' - Encerra o programa")
        print("  • 'limpar' - Limpa a tela")
        print("\n" + "-"*70 + "\n")
        
        while True:
            try:
                # Obter pergunta
                pergunta = input("💬 Você: ").strip()
                
                # Verificar comandos especiais
                if pergunta.lower() in ['sair', 'exit', 'quit', 'q']:
                    print("\n👋 Até logo!")
                    break
                
                if pergunta.lower() in ['limpar', 'clear', 'cls']:
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                if not pergunta:
                    continue
                
                # Processar pergunta
                print("\n🤔 Consultando documento...\n")
                resposta = self.perguntar(pergunta)
                
                # Exibir resposta
                print(f"🤖 Agente: {resposta}\n")
                print("-"*70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrompido pelo usuário. Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}\n")
                continue


# ======================= EXECUÇÃO PRINCIPAL =======================

def main():
    """Função principal"""
    print("\n" + "="*70)
    print("🚀 INICIALIZANDO AGENTE RAG PARA NR-06")
    print("="*70)
    
    try:
        # Verificar argumentos
        recriar = "--recriar" in sys.argv or "--rebuild" in sys.argv
        
        if recriar:
            print("\n⚠️  Modo: RECRIAR vectorstore")
        
        # Criar agente
        agente = AgenteNR06(recriar_vectorstore=recriar)
        
        # Iniciar modo interativo
        agente.modo_interativo()
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
