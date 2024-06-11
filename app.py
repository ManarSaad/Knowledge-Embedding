#import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="Documentation Tracker.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are an automation process agent for searching in a document. 
I will ask you a question and you will give me answer from the document 
and you will follow ALL of the rules below:

1/ Response should be from the the columns in the csv.

2/ You will be asked to count the updatation on the queries and provide the dates, if the updated by is empty it means it is new query and does not count as updated.



{message}

{best_practice}


"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

message="Hello, How many times does Q1 gets updated (count them) ? and please give me the date of the newest update and update by who?"
response=generate_response(message)
print(f'Question: {message},\nAnswer: {response}')
