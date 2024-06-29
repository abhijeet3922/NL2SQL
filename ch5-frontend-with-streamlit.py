OLLAMA_URL = 'http://127.0.0.1:8443'

import streamlit as st
import pandas as pd
import numpy as np
import os
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
from warnings import filterwarnings
filterwarnings('ignore')

import spacy
import torch
import sqlglot
import os, re, logging, pickle
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from sqlglot import parse_one, exp, parse, table, column, to_identifier
from Levenshtein import distance
from sklearn.metrics.pairwise import cosine_similarity
import time

from sqlalchemy import create_engine

def getTableDescriptionSQLiteDB(output_fmt):
    engine = create_engine('sqlite://mysqlitedb.db')
    with engine.connect() as conn, conn.begin():
        sqlite_master = pd.read_sql_query('SELECT * FROM sqlite_master', conn)
    sqlite_master['sql_fmt'] = sqlite_master['sql'].apply(lambda z: [x.strip().strip(',').rsplit(' ', maxsplit=1) for x in z.split('\n')[1:-1]])
    table_desc_dict = {}
    if output_fmt == 'df':
        for _, row in sqlite_master.iterrows():
            table_desc_dict[row['name']] = pd.DataFrame(columns=['name', 'type'], data=row['sql_fmt'])
            table_desc_dict[row['name']]['comment'] = np.nan
    elif output_fmt == 'ddl':
        for _, row in sqlite_master.iterrows():
            table_desc_dict[row['name']] = row['sql']
    return table_desc_dict

def getModelResult(schema, question, model_name, selected_table, table_columns):
    embedding_model_name = 'mixedbread-ai/mxbai-embed-large-v1'
    try:
        print('Running pre-processing...')
        pruned_schema = preprocess_table(question=question, schema=schema, table_name=selected_table)
    except Exception as e:
        print('Pre-processing failed!')
        print(e, e.__traceback__)
        pruned_schema = schema
    
    prompt_template = """### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION] 

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

## Database Schema 
This query will run on a database whose schema is represented in this string: {db_schema} 

### Answer 
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION] [SQL]
"""

    prompt = prompt_template.format(question=question, db_schema=schema)
    print(f'prompt: {prompt}')
    print(f'Querying {model_name}...')
    ollama = OLLAMA(OLLAMA_URL=OLLAMA_URL, model_name='sqlc-7b-2-F16')
    generated_query = ollama.run(prompt)

    if 'i do not know' in generated_query.lower():
        print('Failed to get SQL from model')
        return generated_query, prompt
    print(f'Done! Generated query: {generated_query}')

    print('Running post-processing...')
    try:
        qp = queryPostprocessing(generated_query, {'table_name':selected_table, 'columns':table_columns}, embedding_model_name)
        processed_query = qp.formatQuerySQLglot()
    except Exception as e:
        print('Post-processing failed!')
        print(e, e.__traceback__)
        processed_query = generated_query
    print(f'Done! Processed query: {processed_query}')
    return processed_query, prompt

def getSQLiteDBQueryResult(query):
    engine = create_engine('sqlite:///mysqlitedb.db')
    try:
        with engine.connect() as conn, conn.begin():
            query_result = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f'Error: {e}, {e.__traceback__}')
        query_result = pd.DataFrame()
    return query_result

@st.cache_data
def get_pyg_renderer(df):
    return StreamlitRenderer(df)

######################### MODEL OBJECT DEFINITION BEGINS #########################
import requests
import json

class OLLAMA:
    def __init__(self, OLLAMA_URL, model_name):
        self.model_name = model_name
        self.ollama_url = OLLAMA_URL
        self.ollama_endpoint = '/api/generate'

    def run(self, prompt):
        data = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False
        }

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        resp = requests.post(url = f'{self.ollama_url}{self.ollama_endpoint}',
                             data = json.dumps(data),
                             headers = headers)
        query = resp.json()['response']
        print(f'JSON resp: {query}')
        return query
######################### MODEL OBJECT DEFINITION ENDS #########################

######################### PREPROCESSING MODULES BEGIN #########################
def preprocess_table(question, schema, table_name):
    column_embs, column_descriptions_typed = generate_embeddings(table_name, schema)

    # 1a) get top k columns
    top_k_scores, top_k_indices = knn_(question, column_embs, top_k=20, threshold=0.0)
    topk_table_columns = {}
    table_column_names = set()

    for score, index in zip(top_k_scores, top_k_indices):
        table_name, column_info = column_descriptions_typed[index].split('.', 1)
        column_tuple = re.split(r',\s*(?![^()]*\))', column_info, maxsplit=2) # split only on commas outside parantheses
        if table_name not in topk_table_columns:
            topk_table_columns[table_name] = []
        topk_table_columns[table_name].append(column_tuple)
        table_column_names.add(f'{table_name}.{column_tuple[0]}')
    
    # 1b) get columns which match terms in question
    nlp = spacy.load('en_core_web_trf')
    question_doc = nlp(question)
    q_filtered_tokens = [token.lemma_.lower() for token in question_doc if not token.is_stop]
    q_alpha_tokens = [i for i in q_filtered_tokens if (len(i)>1 and i.isalpha())]

    TIME_TERMS = ['when', 'time', 'hour', 'minute', 'second',
                  'day', 'yesterday', 'today', 'tomorrow',
                  'week', 'month', 'year',
                  'duration', 'date']
    
    time_in_q = False

    nlp_ner = spacy.load('en_core_web_md')
    q_ner_doc = nlp_ner(question)
    ent_types = [w.label_ for w in q_ner_doc.ents]

    if 'DATE' in ent_types or 'TIME' in ent_types:
        time_in_q = True
    elif any([term in question.lower() for term in TIME_TERMS]):
        time_in_q = True
    elif set(q_alpha_tokens).intersection(set(TIME_TERMS)):
        time_in_q = True
    
    for col_details in column_descriptions_typed:
        table_name, column_info = col_details.split('.', 1)
        column_tuple = re.split(r',\s*(?![^()]*\))', column_info, maxsplit=2) # split only on commas outside parantheses
        col_name = column_tuple[0]

        if column_tuple in topk_table_columns[table_name]:
            continue

        # if question concerns time, add time-related columns
        if time_in_q and any([timetype in column_tuple[1] for timetype in ['DATE', 'TIMESTAMP']]):
            if table_name not in topk_table_columns:
                topk_table_columns[table_name] = []
            if column_tuple not in topk_table_columns[table_name]:
                topk_table_columns[table_name].append(column_tuple)
            table_column_names.add(f'{table_name}.{column_tuple[0]}')
            continue

        # if question-token-lemmas overlap with column-token-lemmas, add the column
        column_doc = nlp(col_name.replace('_', ' '))
        col_tokens = [token.lemma_.lower() for token in column_doc if not token.is_stop]
        col_alpha_tokens = [i for i in col_tokens if (len(i)>1 and i.isalpha())]
        if set(col_alpha_tokens).intersection(set(q_alpha_tokens)):
            if table_name not in topk_table_columns:
                topk_table_columns[table_name] = []
            if column_tuple not in topk_table_columns[table_name]:
                topk_table_columns[table_name].append(column_tuple)
            table_column_names.add(f'{table_name}.{column_tuple[0]}')

    # 4) format metadata string
    pruned_schema = format_topk_sql(topk_table_columns, shuffle=False)
    print(f'Pruned schema: {pruned_schema}')
    return pruned_schema

def generate_emebddings(table_name, schema):


######################### PREPROCESSING MODULES BEGIN #########################