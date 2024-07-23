import os
import pickle
import faiss
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Configura a chave da API
os.environ["OPENAI_API_KEY"] = "sk-proj-zSNFInnCxDKmqdJe9LvjT3BlbkFJ5xbp1QwOlM1osPcNhYQM"

# Função para carregar ou criar embeddings
def load_or_create_embeddings(texts, index_path='faiss_index.index', docstore_path='docstore.pkl', id_map_path='index_to_docstore_id.pkl'):
    if os.path.exists(index_path) and os.path.exists(docstore_path) and os.path.exists(id_map_path):
        # Carregar o índice FAISS existente
        index = faiss.read_index(index_path)
        # Carregar o docstore e o mapeamento de IDs
        with open(docstore_path, 'rb') as f:
            docstore = pickle.load(f)
        with open(id_map_path, 'rb') as f:
            index_to_docstore_id = pickle.load(f)
        # Criar a função de embedding
        embedding_function = OpenAIEmbeddings()
        # Criar o FAISS com o índice carregado e a função de embedding
        docsearch = FAISS(index=index, embedding_function=embedding_function, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    else:
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        # Salvar o índice FAISS
        faiss.write_index(docsearch.index, index_path)
        # Salvar o docstore e o mapeamento de IDs
        with open(docstore_path, 'wb') as f:
            pickle.dump(docsearch.docstore, f)
        with open(id_map_path, 'wb') as f:
            pickle.dump(docsearch.index_to_docstore_id, f)
    return docsearch

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # Ler o PDF
        reader = PdfReader('livro.pdf')

        # Ler e concatenar o texto do PDF
        raw_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        # Dividir o texto em chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Carregar ou criar embeddings
        docsearch = load_or_create_embeddings(texts)

        # Configurar o modelo de chat
        llm = OpenAI()  # ou "gpt-3.5-turbo"

        # Criar a cadeia de QA
        chain = load_qa_chain(llm, chain_type="stuff")

        # Consulta
        query = request.query + " -- se a pergunta não tiver no livro, informe que não tem a ver com o tema selecionado"
        docs = docsearch.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
