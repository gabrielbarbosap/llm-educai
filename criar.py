import os
import pickle
import faiss
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Função para carregar ou criar embeddings
def load_or_create_embeddings(texts, index_path, docstore_path, id_map_path):
    if os.path.exists(index_path) and os.path.exists(docstore_path) and os.path.exists(id_map_path):
        print('entrou aqui tem')
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
        print('entrou aqui')
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

def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

def main(query, pdf_path, index_path, docstore_path, id_map_path):
    try:
        # Ler o PDF
        raw_text = read_pdf(pdf_path)

        # Dividir o texto em chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Carregar ou criar embeddings
        docsearch = load_or_create_embeddings(texts, index_path, docstore_path, id_map_path)

        # Configurar o modelo de chat com a chave da API da variável de ambiente
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("Chave API não encontrada.")
        llm = OpenAI(api_key=api_key)  # ou "gpt-3.5-turbo"

        # Criar a cadeia de QA
        chain = load_qa_chain(llm, chain_type="stuff")

        # Consulta
        query = query + " -- se a pergunta não tiver no livro, informe que não tem a ver com o tema selecionado"
        docs = docsearch.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)
        print({"result": result})
    except Exception as e:
        print(f"Erro: {str(e)}")

if __name__ == "__main__":
    query = "Sua pergunta aqui"
    pdf_path = 'historia.pdf'
    index_path = 'fisica2.index'
    docstore_path = 'fisica2.pkl'
    id_map_path = 'fisica2.pkl'
    main(query, pdf_path, index_path, docstore_path, id_map_path)
