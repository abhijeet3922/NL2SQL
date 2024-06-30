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
    engine = create_engine('sqlite:///mysqlitedb.db')
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

    prompt = prompt_template.format(question=question, db_schema=pruned_schema)
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
        qp = queryPostprocessing(generated_query.upper(), {'table_name':selected_table, 'columns':table_columns}, embedding_model_name)
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

def generate_embeddings(table_name, schema):
    num_cols = 0
    TAB_DETAILS = []

    for col in sqlglot.parse_one(schema, dialect='snowflake').find_all(sqlglot.exp.ColumnDef):
        num_cols += 1
        TAB_DETAILS.append([table_name, col.alias_or_name, col.find(sqlglot.exp.DataType).__str__(), col.find(sqlglot.exp.ColumnConstraint)])
    
    encoder = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device='cpu')

    column_descriptions = []
    column_descriptions_typed = []

    for row in TAB_DETAILS:
        tab_name, col_name, col_dtype, col_desc = row
        col_str = f'{tab_name}.{col_name}:{col_desc}'
        col_str_typed = f'{tab_name}.{col_name},{col_dtype},{col_desc}'
        column_descriptions.append(col_str)
        column_descriptions_typed.append(col_str_typed)

    column_embs = encoder.encode(column_descriptions, convert_to_tensor=True, device='cpu')
    return column_embs, column_descriptions_typed

def knn_(query, all_embs, top_k, threshold):
    encoder = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', device='cpu')
    query_emb = encoder.encode(query, convert_to_tensor=True, device='cpu').unsqueeze(0)
    similarity_scores = F.cosine_similarity(query_emb, all_embs)
    top_results = torch.nonzero(similarity_scores > threshold).squeeze()

    # if top_results is empty, return empty tensor
    if top_results.numel() == 0:
        return torch.tensor([]), torch.tensor([])
    
    # if only one result in resturned, convert to tensor
    elif top_results.numel() == 1:
        return torch.tensor([similarity_scores[top_results]]), torch.tensor([top_results])
    
    else:
        top_k_scores, top_k_indices = torch.topk(similarity_scores[top_results], k=min(top_k, top_results.numel()))
        return top_k_scores, top_results[top_k_indices]

def format_topk_sql(topk_table_columns, shuffle):
    if len(topk_table_columns) == 0:
        return ''
    
    md_str = '\n'
    # shuffle the keys in topk_table_columns
    table_names = list(topk_table_columns.keys())
    if shuffle:
        np.random.seed(0)
        np.random.shuffle(table_names)
    for table_name in table_names:
        columns_str = ''
        columns = topk_table_columns[table_name]
        if shuffle:
            np.random.seed(0)
            np.random.shuffle(columns)
        for column_tuple in columns:
            if len(column_tuple) > 2:
                columns_str += (
                    f'\n  {column_tuple[0]} {column_tuple[1]}, --{column_tuple[2]}'
                )
            else:
                columns_str += f'\n  {column_tuple[0]} {column_tuple[1]}, '
        md_str += f'CREATE TABLE {table_name} ({columns_str}\n);\n'
    md_str += '\n'
    return md_str
######################### PREPROCESSING MODULES END #########################

######################### POSTPROCESSING MODULES BEGIN #########################
class queryPostprocessing:

    def __init__(self, query, table_metadata, embedding_model_name):
        self.query = query.upper().split(';')[0]
        self.table_metadata = table_metadata
        self.col_mapping = {}
        self.table_mapping = {}
        self.embedding_model = SentenceTransformer(embedding_model_name, device='cpu')
    
    def getIncorrectColumns(self):
        column_names_query = [col.name for col in parse_one(self.query).find_all(exp.Column)]
        column_names_query = np.unique(column_names_query).tolist()
        column_names_table = self.table_metadata['columns']['name'].tolist()

        matching_cols = list(set(column_names_query).intersection(set(column_names_table)))
        invalid_cols = list(set(column_names_query).difference(set(matching_cols)))
        available_set = list(set(column_names_table).difference(set(matching_cols)))

        # check if enough columns are available to substitute (3 columns in query but only 2 in table)
        for col in matching_cols:
            self.col_mapping[col] = col
        
        for col in invalid_cols:
            closest_col = self.embedding_distance(col, available_set)
            self.col_mapping[col] = closest_col
            available_set.remove(closest_col)
        
        return self.col_mapping
    
    def getIncorrectTables(self):
        ast = parse_one(self.query)
        table_list = []
        for tbl in ast.find_all(exp.Table):
            tbl.set('alias',None)
            table_name = tbl.sql()
            table_list.append(table_name)
        
        unique_tables = np.unique(table_list).tolist()
        if len(unique_tables):
            table_name = unique_tables[0]
            self.table_mapping[table_name] = self.table_metadata['table_name']
        
        return self.table_mapping
    
    def lv_distance(self, col, available_set):
        distances = []
        for column in available_set:
            d = distance(col, column)
            distances.append(d)
        most_similar_column = available_set[np.argmin(distances)]
        return most_similar_column
    
    def embedding_distance(self, col, available_set):
        col_embedding = self.embedding_model.encode([col])
        available_set_embedding = self.embedding_model.encode(available_set)
        similarity_list = cosine_similarity(col_embedding, available_set_embedding)
        most_similar_column = available_set[np.argmax(similarity_list)]
        return most_similar_column
    
    def formatQuery(self):
        _ = self.getIncorrectColumns()
        _ = self.getIncorrectTables()
        updated_query = self.query
        
        for tbl, updated_tbl in self.table_mapping.items():
            updated_query = updated_query.replace(tbl, updated_tbl)
        
        for col, updated_col in self.col_mapping.items():
            updated_query = updated_query.replace(col, updated_col)
        
        return updated_query
    
    def formatQuerySQLglot(self):
        _ = self.getIncorrectColumns()
        _ = self.getIncorrectTables()
        query_ast = parse_one(self.query)
        alias_cols = [al.alias for al in query_ast.find_all(exp.Alias)]

        for col in alias_cols:
            self.col_mapping.pop(col, None)
        
        for tbl in query_ast.find_all(exp.Table):
            table_alias = None
            if 'alias' in tbl.args:
                table_alias = tbl.alias
            tbl.set('alias',None)
            table_name = tbl.sql()
            if table_name in self.table_mapping:
                new_table = table(table=self.table_mapping[table_name], quoted=False, alias=table_alias)
                tbl.replace(new_table)
        
        for col in query_ast.find_all(exp.Column):
            column_name = col.this.this
            if column_name in self.col_mapping:
                col.this.set('this',self.col_mapping[column_name])
        
        return query_ast.sql(dialect='snowflake')
######################### POSTPROCESSING MODULES END #########################

######################### STREAMLIT APP BEGINS #########################
if 'stage' not in st.session_state:
    st.session_state.stage = 0
    print('Deleted existing mysqlite file')
    if os.path.exists('mysqlitedb.db'): os.remove('mysqlitedb.db')

def set_state(i):
    st.session_state.stage = i

if 'df_dict' not in st.session_state:
    st.session_state.df_dict = {}

with st.sidebar:
    st.title('Data Sources')
    source = st.radio('Pick Source', ['CSV', 'Snowflake'], index=None)
    if source == 'CSV':
        with st.popover('Upload CSV'):
            uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'], accept_multiple_files=False, on_change=set_state, args=[0])
            with st.spinner('Processing...'):
                if uploaded_file is not None:
                    filename = uploaded_file.name[:-4].replace('-', '_').replace(' ', '')
                    st.session_state.df_dict[filename] = pd.read_csv(uploaded_file)
                db = create_engine('sqlite:///mysqlitedb.db')
                for key in st.session_state.df_dict:
                    df = st.session_state.df_dict[key]
                    try:
                        df.to_sql(key, db, index=False)
                    except Exception as e:
                        print(f'Failed to load CSV into SQLite DB. Error: {e}')
    elif source == 'Snowflake':
        st.session_state.df_dict = {}
        with st.popover('Fetch from Snowflake'):
            st.write('Left as an exercise to the user...')
    
    model_name = st.radio(label='Model', index=0, options=['sqlc-7b-2-F16', 'GPT-4'])
    model_name = 'sqlc-7b-2-F16'
    st.button(label='Load tables', on_click=set_state, args=[1])

if st.session_state.stage >= 1:
    if source == 'CSV':
        table_desc_dfs = getTableDescriptionSQLiteDB(output_fmt='df')
        for key in table_desc_dfs:
            st.subheader(f'Uploaded CSV Table: {key}')
            st.dataframe(table_desc_dfs[key])
            st.write('\n\n')
        table_desc_ddls = getTableDescriptionSQLiteDB(output_fmt='ddl')
    
    question = st.text_input('Business Query', placeholder='Enter business requirement to be converted to query', on_change=set_state, args=[2])
    st.button('Get SQL Query', on_click=set_state, args=[2])

if st.session_state.stage == 2:
    table_names = list(table_desc_ddls.keys())[0]
    schemas_df = list(table_desc_ddls.values())[0]
    table_columns = list(table_desc_dfs.values())[0]

    with st.spinner('Query generation in progress...'):
        sql_query, prompt = getModelResult(schemas_df, question, model_name, table_names, table_columns)
    st.session_state.sql_query = sql_query
    st.session_state.prompt = prompt
    st.session_state.stage = 3

if st.session_state.stage >= 3:
    sql_query = st.session_state.sql_query
    prompt = st.session_state.prompt
    if sql_query[-1] == ';':
        sql_query = sql_query[:-1]
    
    with st.expander(label='Prompt'):
        st.code(prompt, language='sql')
    
    with st.container():
        modified_query = st.text_area(label=f'Generated query {model_name}', value=sql_query)
    
    st.button('Execute query', on_click=set_state, args=[4])

if st.session_state.stage >= 4:
    try:
        with st.spinner('Fetching data from database...'):
            print('Running SQL on SQLite')
            query_data = getSQLiteDBQueryResult(query=modified_query)
        
        st.dataframe(query_data)

        df_rows, df_cols = query_data.shape
        if df_cols > 1:
            print('Loading pygwalker UI')
            renderer = get_pyg_renderer(query_data)
            renderer.explorer()
        else:
            print('Not loading pygwalker UI for single column result')
        print('SQL executed succesfully!')
        print('Run completed. \n\n')
    except Exception as e:
        st.write(f'Error occured while processing query. {e}, {e.__traceback__}')
        print(f'SQL failed to execute. Error: {e}, {e.__traceback__}')
######################### STREAMLIT APP ENDS #########################