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
from dotenv import load_dotenv

app = FastAPI()

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Função para carregar ou criar embeddings
# Função para carregar ou criar embeddings
def load_or_create_embeddings(texts, index_path, docstore_path, id_map_path):
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
        docsearch = FAISS(
            index=index, 
            embedding_function=embedding_function, 
            docstore=docstore, 
            index_to_docstore_id=index_to_docstore_id
        )
    else:
        print("Arquivos FAISS não encontrados. Criando novos embeddings...")
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        # Salvar o índice FAISS
        faiss.write_index(docsearch.index, index_path)
        # Salvar o docstore e o mapeamento de IDs
        with open(docstore_path, 'wb') as f:
            pickle.dump(docsearch.docstore, f)
        with open(id_map_path, 'wb') as f:
            pickle.dump(docsearch.index_to_docstore_id, f)
        print("Novos embeddings criados e salvos com sucesso.")
    return docsearch
class QueryRequest(BaseModel):
    query: str
    index_path: str
    docstore_path: str
    id_map_path: str

async def read_pdf_async(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text
@app.post("/ask")

async def ask_question(request: QueryRequest):
    try:
        # Ler o PDF de forma assíncrona
        raw_text = await read_pdf_async('livro.pdf')

        # Dividir o texto em chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Definir os caminhos dos arquivos
        index_path = request.index_path
        docstore_path = request.docstore_path
        id_map_path = request.id_map_path

        # Carregar ou criar embeddings
        docsearch = load_or_create_embeddings(texts, index_path, docstore_path, id_map_path)

        # Configurar o modelo de chat com a chave da API da variável de ambiente
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Chave API não encontrada.")
        llm = OpenAI(api_key=api_key)

        # Criar a cadeia de QA
        chain = load_qa_chain(llm, chain_type="stuff")

        # Consulta
        query = "Você é um professor simpatico, cordial e educado. Baseado na sua base de pesquisa que é o que ja foi pre-carregado. você recebera uma pergunta que você deverá entender o contexto dela e saber se ela tem a ver com o tema principal do livro ou se a resposta está no livro. Se ela não tiver a ver nem com o tema principal nem que a resposta esteja exposta de forma clara no livro, você devera respoder o seguinte 'Não tenho a resposta para essa pergunta na minha base de dados', limite-se a responder com no maximo 100 caracteres. Esta foi a pergunta do usuario: " + request.query
        docs = docsearch.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))