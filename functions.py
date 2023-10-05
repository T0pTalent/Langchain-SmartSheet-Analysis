from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
import pinecone
from datetime import date
from config import *


pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

def generate_analysis(openai_key, model_name, phase):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    pinecone_index = pinecone.Index(index_name="all")
    vectorstore = Pinecone(
        index=pinecone_index, embedding=embeddings, text_key="text"
    )
    
    model = ChatOpenAI(
        openai_api_key=openai_key, temperature=0, model_name=model_name
    )
    chain = RetrievalQA.from_chain_type(
        llm=model, chain_type="stuff", retriever=vectorstore.as_retriever()
    )
    
    today = date.today()
    
    prompt = f"""
    % Who you are: 
    - You are a professional excellent Project Manager.

    % What you should know:
    - What you know is all about the one project.
    - Today is {today}.
    - Mustn't consider weekends when calculating date. So You MUST KNOW that a week has 5 workdays when calculating date.

    % What you do:
    - Write your detailed analysis about {phase}.
    - Write current progress of the phase: what to do in the phase, what have been done and what should do in the phase
    - Write what the observation of the project is.
    - Discover potential risk
    - Extract insight
    - Recommendation for future
    """
    
    print(phase)
    
    response = chain.run(prompt)
    
    return response


def generate_answer(openai_key, model_name, phase, query):
    phase_id = phase.split('.')[0]
    
    index_name = 'phase' + phase_id if phase_id.isdigit() else phase
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    
    pinecone_index = pinecone.Index(index_name=index_name)
    vectorstore = Pinecone(
        index=pinecone_index, embedding=embeddings, text_key="text"
    )
    model = ChatOpenAI(
        openai_api_key=openai_key, temperature=0, model_name=model_name
    )
    chain = RetrievalQA.from_chain_type(
        llm=model, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    today = date.today()
    
    prompt = f"""
    You are a professional project manager.
    Answer the following question as best and detailed as possible.
    
    Question: {query}
    """
    
    response = chain.run(prompt)

    print("Query: ", query)
    print('Answer: ', response)
    return response
    