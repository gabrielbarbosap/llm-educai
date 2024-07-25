from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
import os
import pickle
import faiss
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import logging

app = FastAPI()
load_dotenv()

def load_or_create_embeddings(texts, index_path, docstore_path, id_map_path):
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

class QueryRequest(BaseModel):
    query: str
    index_path: str
    docstore_path: str
    id_map_path: str

async def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ''.join([page.extract_text() for page in reader.pages if page.extract_text()])
    return raw_text

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        raw_text = await read_pdf('boleto.pdf')
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_text(raw_text)
        docsearch = load_or_create_embeddings(texts, request.index_path, request.docstore_path, request.id_map_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Chave API não encontrada.")
        llm = OpenAI(api_key=api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        query = "Você é um professor simpático e educado. Quando receber uma pergunta, verifique se a resposta está no livro pré-carregado. Se a resposta estiver no livro, forneça-a de maneira cordial e, se possível, inclua uma breve explicação adicional. Se a resposta não estiver no livro, responda com: 'Não tenho a resposta para essa pergunta na minha base de dados.'. Mantenha suas respostas dentro de 100 caracteres e sempre aja como um professor, promovendo o aprendizado. Essa é a pergunta atual do usuário:" + request.query    
        docs = docsearch.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Initialize logging
logging.basicConfig(level=logging.INFO)

class QuizRequest(BaseModel):
    index_path: str
    docstore_path: str
    id_map_path: str

def are_similar(question, existing_questions, threshold=0.8):
    """ Check if a question is similar to any in the existing questions based on cosine similarity of their TF-IDF vectors """
    if not existing_questions:
        return False
    
    vectorizer = TfidfVectorizer().fit([question] + existing_questions)
    vectors = vectorizer.transform([question] + existing_questions)
    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:])
    
    if similarity_scores.size == 0:
        return False

    return any(score > threshold for score in similarity_scores[0])

@app.post("/create_quiz")
async def create_quiz(request: QuizRequest):
    try:
        # Read and split PDF text
        raw_text = await read_pdf('boleto.pdf')
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_text(raw_text)
        docsearch = load_or_create_embeddings(texts, request.index_path, request.docstore_path, request.id_map_path)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Chave API não encontrada.")
        
        llm = OpenAI(api_key=api_key)
        chain = load_qa_chain(llm, chain_type="stuff")

        # Initial quiz prompt
        quiz_prompt = (
            "Crie uma prova com 10 perguntas complexas sobre 'Os primeiros humanos', baseadas no livro fornecido. Cada pergunta deve ter no mínimo três linhas e deve estimular a interpretação de texto do aluno. Forneça três alternativas para cada pergunta, sendo apenas uma correta. As alternativas devem ser completas e não devem terminar com ponto de interrogação. Não inclua números antes das perguntas."
        )
        
        # Initialize lists and variables
        question_list = []
        quiz_list = []
        num_tries = 0
        max_tries = 5  # Limit the number of retries
        
        while len(quiz_list) < 10 and num_tries < max_tries:
            docs = docsearch.similarity_search(quiz_prompt)
            result = chain.run(input_documents=docs, question=quiz_prompt)
            
            # Process result
            lines = result.split('\n')
            current_question = None
            
            for line in lines:
                line = line.strip()
                if line:
                    if line.endswith('?'):  # Detect if the line is a question
                        if current_question:
                            if are_similar(current_question["question"], question_list):
                                logging.info(f"Pergunta similar encontrada e descartada: {current_question['question']}")
                                current_question = None
                            else:
                                quiz_list.append(current_question)
                                question_list.append(current_question["question"])
                                logging.info(f"Pergunta adicionada: {current_question['question']}")
                                if len(quiz_list) >= 10:
                                    break
                        current_question = {"question": line, "alternatives": []}
                    elif current_question and line[0] in ('a', 'b', 'c'):  # Check if the line starts with 'a', 'b', or 'c'
                        if len(current_question["alternatives"]) < 3:
                            current_question["alternatives"].append(line)
            
            # Check last question
            if current_question and len(current_question["alternatives"]) == 3:
                if not are_similar(current_question["question"], question_list):
                    quiz_list.append(current_question)  # Add the last question if necessary
                    question_list.append(current_question["question"])
                    logging.info(f"Pergunta final adicionada: {current_question['question']}")
                else:
                    logging.info(f"Pergunta final similar encontrada e descartada: {current_question['question']}")
            
            # If less than 10 questions, adjust prompt or take other actions
            if len(quiz_list) < 10:
                logging.info('Tentando novamente...')
                # Optionally modify the prompt or retry logic here
                quiz_prompt = (
                    "Crie uma prova com 10 perguntas complexas sobre 'Os primeiros humanos', baseadas no livro fornecido. Cada pergunta deve ter no mínimo três linhas e deve estimular a interpretação de texto do aluno. Forneça três alternativas para cada pergunta, sendo apenas uma correta. As alternativas devem ser completas e não devem terminar com ponto de interrogação. Não inclua números antes das perguntas."
                )
                
            num_tries += 1
        
        if len(quiz_list) < 10:
            raise HTTPException(status_code=500, detail="Não foi possível gerar 10 perguntas únicas com 3 alternativas.")

        return {"quiz": quiz_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
