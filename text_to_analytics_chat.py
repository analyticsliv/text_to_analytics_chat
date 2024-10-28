#!/usr/bin/env python
# coding: utf-8

# In[1]:

import getpass
import os
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate, MessagesPlaceholder
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
import json
import sys
import google.auth
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from langgraph.graph import START, MessagesState, StateGraph


# In[2]:


def text_to_analytics(input, thread_id, table, chat_llm, llm, db, bq_client):

    class State(TypedDict):
    
        messages: Annotated[Sequence[BaseMessage], add_messages]
        table_info: str

    workflow = StateGraph(state_schema=State)

    def call_model(state: State):
    
        chain = prompt | chat_llm
        response = chain.invoke(state)
        return {"messages": [response]}


    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer = memory)

    def get_sql (input, thread_id, db):
        
        config = {"configurable": {"thread_id": thread_id}}
        query = input
        table_info = db.table_info
        
        input_messages = [HumanMessage(query)]
        output = app.invoke(
            {"messages": input_messages, "table_info": table_info},
            config,
        )
        response = output["messages"][-1].content.replace('```sql', '').replace('```', '')
        response = response.split()
        response = ' '.join(response)
    
        return response
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''You are a Bigquery SQL expert. Given an input question, first create a syntactically correct Bigquery SQL query to run, then look at the results of the query and return the answer to the input question.
    You must query only the columns that are needed to answer the question. Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    
    Along with the SQL query, return the names of the columns used in the query. Remember to return the aliases of the column names, if used. 
    
    DO NOT use backslash (\) to escape characters.
    
    Below are a few examples questions along with their corresponding SQL queries and column names: 
    
    input : Total daily users in September 2024,
    query : SELECT date, active1DayUsers AS Total_Users FROM `wex-ga4-bigquery.wex_nl_to_sql.Active_Users` WHERE date BETWEEN '20240901' AND '20240930',
    Column Names : date, Total_Users
    
    Only use the following tables:
    {table_info}''',
            ),
            MessagesPlaceholder(variable_name = "messages")
        ]
    )
    
    description_prompt = '''
    
    You are expert at annotating datasets. Given a string representation of a dataset, you must do the following:
    
    1) Generate a name for the dataset
    2) Generate a description of the dataset
    3) Generate a description for the fields in the dataset
    4) Give the data types of the fields in the dataset
    
    You must return an updated dictionary without any preamble or explanation.
    
    DO NOT USE CURLY BRACES ANYWHERE IN THE OUTPUT. USE SQUARE BRACES INSTEAD.
    
    Dataset : {input}
    Description : '''
    
    description_prompt = PromptTemplate.from_template(description_prompt)
    describer = description_prompt | llm
    
    viz_prompt = '''
    
    You are an expert data analyst. Given a description of a dataset, your job is to name the two most suitable data visualizations.
    You must return the names of the visualizations without any extra information or explanation. ALWAYS return the output as a list.
    
    DO NOT USE CURLY BRACES IN THE OUTPUT.
    
    Description : {input}
    Visualizations : '''
    
    viz_prompt = PromptTemplate.from_template(viz_prompt)
    visualizer = viz_prompt | llm
    
    prefix = '''
    
    You are a Python data visualization expert who is proficient in Matplotlib. You will be given
    a dataset description and a list of visualizations. Your job is to generate appropriate Matplotlib
    API calls along with the required arguments. Pay attention to the fields information in the dataset descriptiion 
    for arguments. ALWAYS pass data as the value of data argument.
    
    If the Visualizations List contains more than one visualization then generate API calls such that all the listed visualizations
    should be displayed at once.
    
    DO NOT include any import statements. DO NOT include any comments.
    
    YOU MUST NOT INCLUDE ``` AND python in the output.
    
    YOU MUST SEPARATE EACH API CALL WITH ----.
    
    YOU MUST STRICTLY ONLY RETURN THE API CALLS.
    
    Below are some Dataset Descriptions and Visualization Lists along with their corresponding API calls: '''
    
    prefix2 = '''
    
    You are a Python data visualization and Streamlit expert who is proficient in Matplotlib. You will be given
    a dataset description and a list of visualizations. Your job is to generate appropriate Matplotlib
    API calls along with the required arguments. Write the API calls such that they can be displayed on a streamlit UI. Pay attention to the fields information in the dataset descriptiion 
    for arguments. ALWAYS pass data as the value of data argument.
    
    If the Visualizations List contains more than one visualization then generate API calls such that all the listed visualizations
    should be displayed at once.
    
    DO NOT include any import statements. DO NOT include any comments.
    
    YOU MUST NOT INCLUDE ``` AND python in the output.
    
    YOU MUST SEPARATE EACH API CALL WITH ----.
    
    YOU MUST STRICTLY ONLY RETURN THE API CALLS.
    
    Below are some Dataset Descriptions and Visualization Lists along with their corresponding API calls: '''
    
    
    suffix = ''' 
    
    Dataset Description : {input1}
    Visualizations List : {input2}
    API Calls: '''
    
    few_shot_viz = [
    
            {"Dataset Description" : '''"name": "Daily News and Policy Engagement Data",\n"description": "This dataset tracks daily engagement metrics for news and policy content.",\n"fields": [\n"date": "Date in YYYYMMDD format",\n"News": "Number of engagements with news content",\n"Policy": "Number of engagements with policy content"\n],\n"data_types": [\n"date": "int",\n"News": "float",\n"Policy": "float"\n] \n''',
            "Visualizations List" : '["Line chart", "Stacked area chart"] \n',
            "API Calls" : '''a = plt.figure(1)
    plt.plot(data['date'], data['News'], label='News')
    plt.plot(data['date'], data['Policy'], label='Policy')
    plt.xlabel('Date')
    plt.ylabel('Engagements')
    plt.legend()
    plt.show()
    
    ----
    
    b = plt.figure(2)
    plt.stackplot(data['date'], data['News'], data['Policy'], labels=['News', 'Policy'])
    plt.xlabel('Date')
    plt.ylabel('Engagements')
    plt.legend()
    plt.show()'''}
    ]
    
    example_prompt = PromptTemplate.from_template("Dataset Description: {Dataset Description}\nVisualizations List: {Visualizations List}\nAPI Calls: {API Calls}")
    
    function_caller_prompt = FewShotPromptTemplate(
    examples = few_shot_viz,
    example_prompt = example_prompt,
    prefix = prefix,
    suffix = suffix,
    input_variables = ["input1", "input2"])
    
    # function_caller_prompt = PromptTemplate.from_template(function_caller_prompt.format(input1 = description, input2 = visualizations))
    # function_caller = function_caller_prompt | llm

    try:
        
        response = get_sql(input, thread_id, db)

    except:

        st.write('SQL Generation Error.')
    
        # rows = [{'Input' : f'{input}', 'SQL' : 'SQL Generation Error.', 'Data_Description' : '', 'Visualizations' : '', 'API_Calls' : json.dumps({'0' : ''}), 'Remark' : ''}]
        # bq_client.insert_rows_json(table, rows)
    
        st.stop()

    else:
    
        response = response.split()
        response = ' '.join(response)
        sql, column_names = response.split('Column Names')
        sql = sql.replace('SQL Query:', '').replace('SQL query:', '')
        column_names = column_names.replace(': ', '').replace(' ', '').split(',')

    try:
    
        data = pd.DataFrame(eval(db.run(sql)), columns = column_names)

    except:

        st.write('SQL Execution Error')

        # rows = [{'Input' : f'{input}', 'SQL' : f'{sql}', 'Data_Description' : 'SQL Execution Error.', 'Visualizations' : '', 'API_Calls' : json.dumps({'0' : ''}), 'Remark' : ''}]
        # bq_client.insert_rows_json(table, rows)

        st.stop()

    else:
        
        dataframe_string = pd.DataFrame(eval(db.run(sql)), columns = column_names).to_string()

    description = describer.invoke({'input' : dataframe_string})
    visualizations = visualizer.invoke({'input' : description})
    function_caller_prompt = PromptTemplate.from_template(function_caller_prompt.format(input1 = description, input2 = visualizations))
    function_caller = function_caller_prompt | llm
    calls = function_caller.invoke({'input' : 'input'})
    calls = calls.split('----')

    calls_json = {}

    for i in range(len(calls)):

        calls_json[str(i)] = calls[i]

    calls_json = json.dumps(calls_json)

    st.dataframe(data)

    #try:
    st.write('Here')

    #import matplotlib.pyplot as plt

    for call in calls:

        st.write('Got here')

        exec(call)

    # except:

    #     st.write('API Call Execution Error')

        # rows = [{'Input' : f'{input}', 'SQL' : f'{sql}', 'Data_Description' : f'{description}', 'Visualizations' : f'{visualizations}', 'API_Calls' : f'{calls_json}', 'Remark' : ''}]
        # bq_client.insert_rows_json(table, rows)

        st.stop()

    return sql, description, visualizations


# In[7]:




st.set_page_config(page_title = 'AI Dashboard - Washington Examiner. (Testing Interface)')
st.header('AI Dashboard - Washington Examiner. (Testing Interface)\n\nThis application is powered by Gemini Pro')

def click_button():
    
    st.session_state.clicked = True

project = 'wex-ga4-bigquery'
dataset = 'wex_nl_to_sql'

url = f'bigquery://{project}/{dataset}'#credentials_path={credentials_path}'
db = SQLDatabase.from_uri(url)

bq_client = bigquery.Client()
table = bq_client.get_table("dx-api-project.text_to_analytics_chat.testing")
users_table = bq_client.get_table("dx-api-project.text_to_analytics_chat.Users")

#data_ana_key = 'AIzaSyBNJ5yDbSIPLTKlB-fcSPDrL95hM94sppE'
chat_llm = ChatVertexAI(model = "gemini-1.5-pro-latest", temperature = 0.1)
llm = VertexAI(model = "gemini-1.5-pro-latest", temperature = 0.1)


username = st.text_input("Name: ", key = 'name')

if 'clicked' not in st.session_state:
    
    st.session_state.clicked = False
    
get_name = st.button('Submit your name', key = 'username', on_click = click_button)

if st.session_state.clicked:

    query = f'SELECT * FROM `dx-api-project.text_to_analytics_chat.Users` WHERE Name = "{username}"'
    query_job = bq_client.query(query)
    users_data = [(row.Name, row.Thread_ID) for row in query_job.result()]
    

    if len(users_data) != 0: #username in list(query_job.iloc[: , 0]):

        thread_id = users_data[-1][1] #query_job.iat[-1, 1
        st.write(f'Nice to see you again, {username}!')

    else:

        st.write('It seems you are new here. Nice to see you!')
        get_new_name = st.button('Submit your name', key = 'new_username')
    
        if get_new_name:
    
            thread_id = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(5))
            rows = [{'Name' : username, 'Thread_ID' : thread_id}]
            bq_client.insert_rows_json(users_table, rows)


    sql_ = []
    description_ = []
    visualizations_ = []
    remark_ = []

    input = st.text_input("Input: ", key = 'input')
    submit = st.button("Ask a question", key = 'submit_input')#, on_click = click_button)
    session_end = st.button("End Session", key = 'end_session')
    
    if session_end:

        sql_json = {}
        description_json = {}
        visualizations_json = {}
        remark_json = {}

        for i in range(len(sql_)):

            sql_json[i] = sql_[i]
            description_json[i] = description_[i]
            visualizations_json[i] = visualizations_[i]
            remark_json[i] = remark_[i]

            rows = [{'Name' : username, 'Thread_ID' : thread_id, 'Input' : input, 'SQL' : sql_json, 'Description' : 'description_json', 'Visualizations' : isualizations_json, 'Remark' : remark_json}]

    
    if submit:
        
        sql, description, visualizations = text_to_analytics (input, thread_id, table, chat_llm, llm, db, bq_client)

        sql_.append(sql)
        description_.append(description)
        visualizations_.append(visualizations)
        
        remark = st.text_input("Remark: ", key = 'remark')
        submit_remark = st.button("Please submit a remark")

        if submit_remark:

            remark_.append(remark)
