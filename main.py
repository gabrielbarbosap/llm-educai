from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
import pickle
import faiss
import logging
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI

app = FastAPI()
load_dotenv()

# Inicializa o logging
logging.basicConfig(level=logging.INFO)

def load_or_create_embeddings(texts, index_path, docstore_path, id_map_path):
    """
    Carrega ou cria embeddings a partir dos textos fornecidos.
    Se os arquivos de índice e de armazenamento de documentos existirem, eles são carregados.
    Caso contrário, um novo índice é criado e salvo.
    """
    if os.path.exists(index_path) and os.path.exists(docstore_path) and os.path.exists(id_map_path):
        index = faiss.read_index(index_path)
        with open(docstore_path, 'rb') as f:
            docstore = pickle.load(f)
        with open(id_map_path, 'rb') as f:
            index_to_docstore_id = pickle.load(f)
        embedding_function = OpenAIEmbeddings()
        docsearch = FAISS(index=index, embedding_function=embedding_function, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    else:
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        faiss.write_index(docsearch.index, index_path)
        with open(docstore_path, 'wb') as f:
            pickle.dump(docsearch.docstore, f)
        with open(id_map_path, 'wb') as f:
            pickle.dump(docsearch.index_to_docstore_id, f)
    return docsearch

async def read_pdf(pdf_path):
    """
    Lê o texto de um PDF e retorna o texto extraído.
    """
    reader = PdfReader(pdf_path)
    raw_text = ''.join([page.extract_text() for page in reader.pages if page.extract_text()])
    return raw_text

class QueryRequest(BaseModel):
    """
    Modelo de solicitação para consultar o PDF.
    """
    query: str
    index_path: str
    docstore_path: str
    id_map_path: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Endpoint para responder perguntas com base no conteúdo do PDF.
    """
    try:
        # Lê o texto do PDF
        raw_text = await read_pdf('boleto.pdf')
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_text(raw_text)
        # Carrega ou cria embeddings
        docsearch = load_or_create_embeddings(texts, request.index_path, request.docstore_path, request.id_map_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Chave API não encontrada.")
        llm = OpenAI(api_key=api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        query = "Você é um professor simpático e educado. Quando receber uma pergunta, verifique se a resposta está no livro pré-carregado. Se a resposta estiver no livro, forneça-a de maneira cordial e, se possível, inclua uma breve explicação adicional. Se a resposta não estiver no livro, responda com: 'Não tenho a resposta para essa pergunta na minha base de dados.'. Mantenha suas respostas dentro de 100 caracteres e sempre aja como um professor, promovendo o aprendizado. Essa é a pergunta atual do usuário:" + request.query
        # Busca os documentos relevantes
        docs = docsearch.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QuizRequest(BaseModel):
    """
    Modelo de solicitação para criar um quiz.
    """
    prompt: str  
    dificulty: str
    year: str
    content: str
    vocabulary: str
    goal: str
    qtd_questions: int 

@app.post("/create_quiz")
async def create_quiz(request: QuizRequest):
    """
    Endpoint para criar um quiz com base nas especificações fornecidas.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Chave API não encontrada.")
        
        # Inicializa o cliente OpenAI com a chave API
        client = OpenAI(api_key=api_key)
        
        quiz_prompt = (
            f"Crie uma prova com 10 perguntas sobre o tema: '{request.content}',"
            f" baseadas no ano que irei informar e atente-se a atender a Base Nacional Comum Curricular. As perguntas deverão ser com o objetivo de estimular {request.goal} dos alunos."
            f"Sobre as perguntas: use um vocabulário {request.vocabulary}, elas terão um nível de dificuldade {request.dificulty} e será respondida por alunos do {request.year} ano."
            f"Informações adicionais: {request.prompt}."
            " Forneça três alternativas(a,b,c) para cada pergunta, sendo apenas uma correta."
            " Toda pergunta deverá terminar com um ponto de interrogação."
            " As alternativas devem ser completas e não devem terminar com ponto de interrogação."
        )
        quiz_list = []
        num_tries = 0
        max_tries = 10
        
        while len(quiz_list) < 10 and num_tries < max_tries:
            try:

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Substitua pelo modelo adequado, como gpt-4
                    messages=[
                        {"role": "system", "content": "Você é um assistente de criação de desafios/quizes."},
                        {"role": "user", "content": quiz_prompt}
                    ],
                    max_tokens=3000,  # Ajuste conforme necessário
                    temperature=0.7
                )

                result = response.choices[0].message.content.strip()
                if not result:
                    logging.warning("Resposta vazia recebida da API.")
                    raise HTTPException(status_code=500, detail="Resposta vazia da API.")
                
                lines = result.split('\n')
                current_question = None
                
                for line in lines:
                    line = line.strip()
                    if line:
                        if line.endswith('?'):
                            if current_question and len(current_question["alternatives"]) == 3:
                                quiz_list.append(current_question)
                                if len(quiz_list) >= 10:
                                    break
                            current_question = {"question": line, "alternatives": []}
                        elif current_question and len(current_question["alternatives"]) < 3 and line[0] in ('a', 'b', 'c'):
                            current_question["alternatives"].append(line)
                
                if current_question and len(current_question["alternatives"]) == 3:
                    quiz_list.append(current_question)
                
                if len(quiz_list) < 10:
                    logging.info('Tentando novamente...')
                
                num_tries += 1
            
            except Exception as e:
                logging.error(f"Erro ao gerar perguntas: {e}")
                raise HTTPException(status_code=500, detail="Erro ao gerar perguntas.")
        
        if len(quiz_list) < 10:
            raise HTTPException(status_code=500, detail="Não foi possível gerar 10 perguntas únicas com 3 alternativas.")
        
        return {"quiz": quiz_list}
    
    except Exception as e:
        logging.error(f"Erro ao gerar perguntas: {e}")
        raise HTTPException(status_code=500, detail=str(e))
