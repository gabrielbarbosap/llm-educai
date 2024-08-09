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

class ScheduleRequest(BaseModel):
    topic: str
    details: str
    class_duration: str
    additional: str
    number_of_classes: int

class ScheduleItem(BaseModel):
    lesson_number: int
    topic: str
    duration: str
    suggestions: str

class ScheduleResponse(BaseModel):
    schedule: List[ScheduleItem]


class QuizRequest(BaseModel):
    content: str
    goal: str
    vocabulary: str
    difficulty: str
    year: str
    additional: str

class QueryRequest(BaseModel):
    """
    Modelo de solicitação para consultar o PDF.
    """
    query: str
    index_path: str
    docstore_path: str
    id_map_path: str
    
class StoryRequest(BaseModel):
    subject: str
    key_concepts: str
    audience_age: int
    style: str
    length: int

class StoryResponse(BaseModel):
    story: str

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
            f"Crie uma prova de dificuldade {request.difficulty} com exatamente 10 perguntas (é fundamental que você retorne sempre o número exato de perguntas.) sobre o tema '{request.content}'. "
            f"Baseie-se no ano do aluno que é: {request.year}. E siga a Base Nacional Comum Curricular. As perguntas devem estimular {request.goal} dos alunos. "
            f"Utilize um vocabulário {request.vocabulary} e um nível de dificuldade {request.difficulty} nas perguntas. "
            f"As perguntas devem ser no formato do ENEM, isto é, complexas e desafiadoras. Inclua três alternativas (a, b, c), sendo apenas uma correta. "
            f"As perguntas devem terminar com um ponto de interrogação(isto é fundamental). As alternativas devem ser completas e não terminar com ponto de interrogação(isto é fundamental). "
            f"Identifique a resposta correta com a etiqueta 'Resposta Correta:' e indique a letra da alternativa correta (a, b, c). "
            f"Busque não repetir nenhuma pergunta, isso é fundamental. Alem disso, siga essas instruções adicionais: {request.additional}"
            f"Evite usar negrito ou qualquer formatação adicional. "
            f"Exemplo de formato a ser sempre seguido: Pergunta? \n a) Alternativa 1 \n b) Alternativa 2 \n c) Alternativa 3 \n Resposta Correta: a"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em criar provas para alunos de ensino médio."},
                {"role": "user", "content": quiz_prompt}
            ],
            max_tokens=1500,
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
                if len(quiz_list) >= 10:
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
        
        if len(quiz_list) < 10:
            logging.warning(f"Foi gerado menos de 5 perguntas. Foram geradas {len(quiz_list)} perguntas.")
        
        return {"quiz": quiz_list, "warning": "Menos de 5 perguntas foram geradas." if len(quiz_list) < 10 else None}
    
    except Exception as e:
        logging.error(f"Erro ao gerar perguntas: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/create_schedule", response_model=ScheduleResponse)
async def create_schedule(request: ScheduleRequest): 
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Chave API não encontrada.")
        
        client = OpenAI(api_key=api_key)
        
        schedule_prompt = (
            f"Crie um cronograma de aula detalhado com base nos seguintes tópicos e informações: \n\n"
            f"Tópico: {request.topic}\n"
            f"Detalhes: {request.details}\n"
            f"Duração das Aulas: {request.class_duration}\n"
            f"Instruções Adicionais: {request.additional}\n"
            f"Número de Aulas: {request.number_of_classes}\n\n"
            f"O cronograma deve incluir a numeração das aulas, tópicos a serem abordados em cada aula, "
            f"e a duração de cada aula. Inclua sugestões de como conduzir a aula, como atividades ou discussões. "
            f"Evite usar formatação adicional e siga o formato abaixo:\n\n"
            f"Aula: Número da aula\n"
            f"Tópico: Descrição do tópico\n"
            f"Duração: Tempo estimado\n"
            f"Sugestões: Atividades ou tópicos para discussão\n\n"
            f"Exemplo:\n"
            f"Aula: 1\n"
            f"Tópico: Introdução à Física\n"
            f"Duração: 60 minutos\n"
            f"Sugestões: Discutir conceitos básicos e fazer uma demonstração prática\n\n"
            f"Crie um cronograma que seja detalhado e organizado, facilitando o planejamento das aulas."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Ou outro modelo adequado
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em criar cronogramas de aula detalhados."},
                {"role": "user", "content": schedule_prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        if not result:
            raise HTTPException(status_code=500, detail="Resposta vazia recebida da API.")
        
        # Processa a resposta para garantir o formato adequado
        schedule_lines = result.split('\n')
        schedule_list = []
        
        current_item = {}
        for line in schedule_lines:
            line = line.strip()
            if line.startswith("Aula:"):
                if current_item:
                    schedule_list.append(ScheduleItem(**current_item))
                current_item = {"lesson_number": int(line.split("Aula:")[1].strip())}
            elif line.startswith("Tópico:"):
                current_item["topic"] = line.split("Tópico:")[1].strip()
            elif line.startswith("Duração:"):
                current_item["duration"] = line.split("Duração:")[1].strip()
            elif line.startswith("Sugestões:"):
                current_item["suggestions"] = line.split("Sugestões:")[1].strip()
        
        # Adiciona o último item
        if current_item:
            schedule_list.append(ScheduleItem(**current_item))
        
        if len(schedule_list) != request.number_of_classes:
            raise HTTPException(status_code=500, detail="Número de aulas gerado não corresponde ao número solicitado.")
        
        if not schedule_list:
            raise HTTPException(status_code=500, detail="Cronograma não gerado corretamente.")
        
        return {"schedule": schedule_list}
    
    except Exception as e:
        logging.error(f"Erro ao gerar cronograma: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_story", response_model=StoryResponse)
async def create_story(request: StoryRequest):
    """
    Endpoint para criar uma história lúdica para ensinar um conteúdo com base nos parâmetros fornecidos.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Chave API não encontrada.")
        
        client = OpenAI(api_key=api_key)
        
        story_prompt = (
            f"Crie uma história curta e educativa para ensinar o seguinte conteúdo:\n\n"
            f"Assunto: {request.subject}\n"
            f"Conceitos-chave: {request.key_concepts}\n"
            f"Idade do Público-Alvo: {request.audience_age}\n"
            f"Estilo: {request.style}\n"
            f"Duração da História: Aproximadamente {request.length} minutos\n\n"
            f"A história deve ser clara, envolvente e adequada para a idade especificada. "
            f"Faça com que os conceitos-chave sejam apresentados de maneira natural e divertida. "
            f"Evite explicações longas e mantenha a narrativa direta ao ponto.\n\n"
            f"Exemplo de resposta esperada:\n"
            f"Era uma vez um herói que descobriu um novo conceito de física e usou-o para salvar o mundo. "
            f"Ele enfrentou desafios e fez descobertas sobre como a força e a massa influenciam a aceleração. "
            f"No final, ele explica o conceito de forma simples para o público.\n\n"
            f"Crie uma história que seja educativa e cativante."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Ou outro modelo adequado
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em criar histórias educativas e envolventes."},
                {"role": "user", "content": story_prompt}
            ],
            max_tokens=1000,  # Ajuste conforme necessário
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        if not result:
            raise HTTPException(status_code=500, detail="Resposta vazia recebida da API.")
        
        return {"story": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar a história: {e}")
    
class Question(BaseModel):
    question: str
    selected_answer: str
    correct_answer: str

class AnswerRequest(BaseModel):
    questions: list[Question]

@app.post("/grade_quiz")
async def grade_quiz(request: AnswerRequest):
    """
    Endpoint para corrigir várias perguntas e fornecer feedback com base nas respostas fornecidas.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Chave API não encontrada.")
        
        client = OpenAI(api_key=api_key)

        # Formatação do prompt para o modelo
        questions_str = "\n".join([
            f"Pergunta: {q.question}\nResposta Marcada: {q.selected_answer}\nResposta Correta: {q.correct_answer}"
            for q in request.questions
        ])

        prompt = (
            "Analise o quiz a seguir e identifique quais respostas estão corretas e quais estão erradas. "
            "Comporte-se como um professor gentil e motivador, oferecendo feedback que ajude o aluno a crescer. "
            "Sua resposta deve seguir este padrão: "
            "'Você acertou as questões: (número das questões corretas(APENAS O NUMERO))' "
            "'Sobre seus erros: (questões erradas e dicas de melhoria detalhadas, oferecendo sugestões de estudo e explicações claras para ajudar no entendimento)' "
            "'Dicas de estudo: (sugestões construtivas e encorajadoras sobre como aprofundar o conhecimento, como recursos recomendados e estratégias de estudo eficazes)'"
            "Na seção sobre seus erros siga esse esse exemplo: "
            "- Questão 3: A capital do Brasil é Brasília, e não o Rio de Janeiro. É importante lembrar que Brasília foi inaugurada como a nova capital em 1960, com o objetivo de promover o desenvolvimento do interior do país. Uma dica é revisar a história das capitais brasileiras e as razões que levaram à mudança para Brasília."
            "Na seção de dicas de estuda siga esse esse exemplo: "
            "Dicas de estudo:\n"
            "- Para aprofundar seu conhecimento sobre a Primeira Guerra Mundial, você pode ler livros como \"A Primeira Guerra Mundial\" de John Keegan ou \"A Grande Guerra\" de Paul Fussell. Esses autores oferecem uma visão abrangente e detalhada do conflito.\n"
            "- Assistir a documentários ou séries sobre a Primeira Guerra Mundial também pode ser muito útil. O documentário \"The Great War\" da PBS é uma excelente opção.\n"
            "- Além disso, considere fazer resumos e mapas mentais sobre os principais eventos e tratados da guerra, isso pode ajudar a fixar melhor as informações. Lembre-se, cada erro é uma oportunidade de aprendizado, e você está no caminho certo! Continue assim!"
            "Certifique-se de ser encorajador e mostrar que erros são oportunidades para aprender. "
            "Mantenha o feedback claro, positivo e útil, com informações que inspirem e guiem o aluno."
            f"\n\n{questions_str}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em corrigir perguntas e fornecer feedback detalhado."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        if not result:
            raise HTTPException(status_code=500, detail="Resposta vazia recebida da API.")
        
        # Estrutura da resposta
        feedback = {
            "acertos": [],
            "erros": [],
            "dicas_de_estudo": []
        }
        print(result)
        # Separar seções usando linhas e padrões específicos
        lines = result.split('\n')
        section = None
        
        for line in lines:
            print(line)
            line = line.strip()
            if line.startswith("Você acertou as questões:"):
                section = "acertos"
                feedback["acertos"] = [q.strip() for q in line[len("Você acertou as questões:"):].strip().split(",") if q.strip()]
            elif line.startswith("Sobre seus erros:"):
                section = "erros"
                feedback["erros"] = []
            elif line.startswith("Dicas de estudo:"):
                section = "dicas_de_estudo"
                feedback["dicas_de_estudo"] = []
            elif section == "erros":
                if line:
                    feedback["erros"].append(line)
            elif section == "dicas_de_estudo":
                if line:
                    feedback["dicas_de_estudo"].append(line)
        
        # Se não houver erros, criar uma lista vazia
        if not feedback["erros"]:
            feedback["erros"] = []
        if not feedback["dicas_de_estudo"]:
            feedback["dicas_de_estudo"] = []

        return feedback
    
    except Exception as e:
        logging.error(f"Erro ao corrigir as perguntas: {e}")
        raise HTTPException(status_code=500, detail=str(e))