import streamlit as st

import requests 
import minsearch

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)

def build_prompt(query, search_results):
    prompt_template = """
QUESTION: {question}

CONTEXT: 
{context}

ANSWER:
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"{doc['question']}\n{doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

def llm(prompt):
    response = client.chat.completions.create(
        model='phi3',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

from elasticsearch import Elasticsearch
es_client = Elasticsearch('http://localhost:9200')

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}

index_name = "course-questions"

#es_client.indices.create(index=index_name, body=index_settings)

#for doc in documents:
#    es_client.index(index=index_name, document=doc)

def search(query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    result_docs = []
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    print(result_docs)
    return result_docs

# Define the 'rag' function which takes an input and returns a result
def rag(input_value):
    search_results = search(input_value)
    prompt = build_prompt(input_value, search_results)
    answer = llm(prompt)
    return answer

# Create a Streamlit app with an input box and a button
st.title('RAG Function Executor')

# Create a form for the input and button
with st.form(key='rag_form'):
    input_value = st.text_input("Enter your input")
    submit_button = st.form_submit_button(label='Ask')

# When the button is clicked
if submit_button:
    # Show a loading message
    with st.spinner('Processing...'):
        # Invoke the 'rag' function
        result = rag(input_value)
    # Display the result
    st.success(f'Result: {result}')
