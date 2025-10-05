# RAG Agent para NR-06

Sistema de perguntas e respostas sobre a Norma Regulamentadora NR-06 (Equipamentos de Proteção Individual) usando RAG (Retrieval Augmented Generation).

## Requisitos

- Python 3.9+
- Ollama instalado e rodando
- 8GB RAM (mínimo)

## Instalação

```bash
# Instalar dependências
pip install -r requirements.txt

# Instalar e iniciar Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3
ollama serve
```

## Como usar

```bash
python rag.py
```

O sistema vai:
1. Carregar o PDF da NR-06
2. Criar embeddings e indexar no FAISS (primeira vez leva alguns minutos)
3. Abrir interface para perguntas

## Estrutura

```
.
├── rag.py                  # Script principal
├── requirements.txt        # Dependências Python
├── data/
│   └── nr-06-atualizada-2022-1.pdf
└── reference_files/        # Arquivos de referência
```

## Tecnologias

- **LangChain**: orquestração do RAG
- **Ollama**: LLM local (llama3)
- **FAISS**: busca vetorial
- **HuggingFace**: embeddings multilíngues

## Configuração

Edite as variáveis na classe `Config` do `rag.py`:

```python
OLLAMA_MODEL = "llama3"           # Modelo a usar
CHUNK_SIZE = 1200                 # Tamanho dos chunks
TOP_K = 4                         # Chunks a recuperar
USE_RERANKER = True               # Ativar reranking
```

## Exemplos de perguntas

- Qual o objetivo da NR-06?
- Quais são as obrigações do empregador?
- O que é o Certificado de Aprovação?
- Quais EPIs protegem a cabeça?

## Notas

- Na primeira execução, o vectorstore é criado e salvo em `vectorstore/`
- Use `--recriar` para reprocessar o documento
- O servidor Ollama precisa estar rodando em `http://localhost:11434`

## Licença

Uso acadêmico e educacional.
