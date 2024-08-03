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
    content: str
    goal: str
    vocabulary: str
    difficulty: str
    year: str
    additional: str

@app.post("/create_quiz")
async def create_quiz(request: QuizRequest):
    """
    Endpoint para criar um quiz com base nas especificações fornecidas.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Chave API não encontrada.")
        
        client = OpenAI(api_key=api_key)
        
        quiz_prompt = (
            f"Crie uma prova de dificuldade {request.difficulty} com exatamente 5 perguntas (é fundamental que você retorne sempre o número exato de perguntas.) sobre o tema '{request.content}'. "
            f"Baseie-se no ano do aluno que é: {request.year}. E siga a Base Nacional Comum Curricular. As perguntas devem estimular {request.goal} dos alunos. "
            f"Utilize um vocabulário {request.vocabulary} e um nível de dificuldade {request.difficulty} nas perguntas. "
            f"As perguntas devem ser no formato do ENEM, isto é, complexas e desafiadoras. Inclua três alternativas (a, b, c), sendo apenas uma correta. "
            f"As perguntas devem terminar com um ponto de interrogação(isto é fundamental). As alternativas devem ser completas e não terminar com ponto de interrogação(isto é fundamental). "
            f"Identifique a resposta correta com a etiqueta 'Resposta Correta:' e indique a letra da alternativa correta (a, b, c). "
            f"Evite usar negrito ou qualquer formatação adicional. "
            f"Exemplo de formato a ser sempre seguido: Pergunta? \n a) Alternativa 1 \n b) Alternativa 2 \n c) Alternativa 3 \n Resposta Correta: a"
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em criar provas para alunos de ensino médio."},
                {"role": "user", "content": quiz_prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        if not result:
            raise HTTPException(status_code=500, detail="Resposta vazia recebida da API.")
        
        lines = result.split('\n')
        quiz_list = []
        current_question = None
        correct_answer_letter = None
        
        for line in lines:
            line = line.strip()
            if line.endswith('?'):
                if current_question and len(current_question["alternatives"]) == 3:
                    if correct_answer_letter:
                        # Adiciona a resposta correta
                        current_question["correct_answer"] = correct_answer_letter
                        quiz_list.append(current_question)
                    else:
                        logging.error("Resposta correta não encontrada para a pergunta.")
                if len(quiz_list) >= 5:
                    break
                current_question = {"question": line, "alternatives": [], "correct_answer": ""}
                correct_answer_letter = None
            elif current_question and len(current_question["alternatives"]) < 3 and line[0] in ('a', 'b', 'c'):
                current_question["alternatives"].append(line[3:].strip())  # Remove o prefixo (a, b, c) e espaços extras
            elif line.startswith("Resposta Correta:"):
                correct_answer_letter = line[len("Resposta Correta:"):].strip()
        
        if current_question and len(current_question["alternatives"]) == 3:
            if correct_answer_letter:
                current_question["correct_answer"] = correct_answer_letter
                quiz_list.append(current_question)
        
        if len(quiz_list) < 5:
            logging.warning(f"Foi gerado menos de 5 perguntas. Foram geradas {len(quiz_list)} perguntas.")
        
        return {"quiz": quiz_list, "warning": "Menos de 5 perguntas foram geradas." if len(quiz_list) < 5 else None}
    
    except Exception as e:
        logging.error(f"Erro ao gerar perguntas: {e}")
        raise HTTPException(status_code=500, detail=str(e))