import smartsheet
import re
import pandas
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
import pinecone
from config import *


def contains_phase_number(text):
    # pattern = r"phase\s*\d"
    pattern = r"phase\s"
    
    return bool(re.search(pattern, text.lower()))


def preprocess_all_conent(access_token, sheet_id):
    smart_client = smartsheet.Smartsheet(access_token=access_token)
    current_sheet = smart_client.Sheets.get_sheet(sheet_id)
    columns = smart_client.Sheets.get_columns(sheet_id).data
    
    headers = [c.title for c in columns]
    
    column_ids = [[c for c in columns if c.title == h][0].id for h in headers]
    
    raw_text = ""
    temp_text = ""
    phase_id = 1
    
    phases = []
    individual_phases = []
    
    pre_available = False
    
    for row in current_sheet.rows:
        values = [
            [cell.value for cell in row.cells if cell.column_id == col][0]
            for col in column_ids
        ]
        if values[headers.index('Task Name')] is not None and contains_phase_number(values[headers.index('Task Name')]):
            raw_text += '\n\n' + '-' * 30 + f' {phase_id}. ' + values[headers.index('Task Name')] + ' ' + '-'*30 +'\n'
            phases.append(f' {phase_id}. ' + values[headers.index('Task Name')])
            phase_id += 1
            
            individual_phases.append(temp_text)
            temp_text = '\n\n' + '-' * 30 + ' ' + values[headers.index('Task Name')] + ' ' + '-'*30 +'\n'
        for i in range(len(values)):
            if values[i] != None:
                raw_text += headers[i] + ": " + str(values[i]) + "\n"
                temp_text += headers[i] + ": " + str(values[i]) + "\n"
        raw_text += "\n"
        temp_text += '\n'
    
    individual_phases.append(temp_text)
    
    with open("processed_files/phases.txt", "w") as phase_file:
        phase_file.write('\n'.join(phases))
        
    with open("processed_files/all_content.txt", "w") as raw_file:
        raw_file.write(raw_text)
        
    if individual_phases[0] is not None and len(individual_phases[0]) > 1000:
        pre_available = True
        with open("processed_files/pre.txt", "w") as pre_file:
            pre_file.write(individual_phases[0])
            
    for i in range(1, len(individual_phases)):
        with open(f"processed_files/phase{i}.txt","w") as file:
            file.write(individual_phases[i])
            
    print('Preprocess Done!!!')
    
    return pre_available
        
        
def delete_index():
    for index in pinecone.list_indexes():
        pinecone.delete_index(index)
    
    print('Deleted All Indexes')
        
        
def build_pinecone_index(filename, index_name, chunck_size, openai_key):
    loader = TextLoader(f"processed_files/{filename}.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunck_size, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric="dotproduct", dimension=1536)
        vectorstore = Pinecone.from_documents(
            docs, embedding=embeddings, index_name=index_name
        )
        
        print('Created Index:', index_name)


def build_knowledge_base(access_token, sheet_id, openai_key, model_name):
    pre_available = preprocess_all_conent(access_token, sheet_id)
        
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    
    delete_index()
    
    build_pinecone_index("all_content", 'all', 450, openai_key)
    
    if pre_available:
        build_pinecone_index('pre', 'baseline', 450, openai_key)
    
    phases = [s.strip() for s in open('processed_files/phases.txt', 'r').readlines()]
    
    for i in range(len(phases)):
        build_pinecone_index(f'phase{i+1}', f'phase{i+1}', 450, openai_key)
        
    if pre_available:
        phases = ['all', 'baseline'] + phases
    else:
        phases = ['all'] + phases
        
    with open("processed_files/phases.txt", "w") as phase_file:
        phase_file.write('\n'.join(phases))
        
    return phases
        