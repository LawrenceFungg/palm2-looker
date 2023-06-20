import json
import traceback
import streamlit as st
import streamlit.components.v1 as components
import os
from time import sleep
from langchain.document_loaders import DirectoryLoader

from google.cloud import aiplatform

# from langchain.llms import OpenAI
from langchain.llms import VertexAI
import looker_sdk
from indexer import Indexer
from looker_query_runner import LookerQueryRunner

from query_converter import QueryConverter
from looker_look_creator import LookerLookCreator

sdk = looker_sdk.init40("looker.ini")
model_name = os.environ.get("LOOKER_MODEL_NAME")
lookml_dir = os.environ.get("LOOKML_DIR")

# error if the model name and lookml_dir is not set
if model_name is None or lookml_dir is None:
    raise Exception("Please set LOOKER_MODEL_NAME and LOOKML_DIR")

aiplatform.init(project="cloud-llm-preview1", location="us-central1")
llm = VertexAI(max_output_tokens=1024)
# llm = OpenAI(model_name="text-davinci-003", temperature=0)


@st.cache_resource
def create_index():
    loader = DirectoryLoader(lookml_dir)
    docs = loader.load()

    return Indexer(docs, llm).run()


docsearch = create_index()

query_converter = QueryConverter(model_name, docsearch, llm)

looker_query_runner = LookerQueryRunner(sdk)
looker_look_creator = LookerLookCreator(sdk)
available_chart_types = ['looker_column' ,'looker_bar','looker_line','looker_scatter','looker_area','looker_pie','single_value','looker_grid']

# This example shows how to make a simple question in natural language and get the result back from the Looker API.
looker_app_title = os.environ.get("LOOKER_APP_TITLE")

# error if the app name is not set
if looker_app_title is None:
    raise Exception("Please set LOOKER_APP_TITLE")

st.title(f"{looker_app_title}")

text = "Show me a list of the names and id of the products."

question = st.text_area("Question", placeholder=text)

if st.button("Send") or question:
    try:
        with st.spinner('Wait for it...'):
            write_query = query_converter.run(question)
            st.markdown("## Query")
            st.markdown(f"```json\n{json.dumps(write_query, indent=2)}\n```")

            st.markdown("## Looker Visualisation")
            look = looker_look_creator.create_look(
                query=write_query, 
                chart_types=available_chart_types
            )
            print(look['embed_url'])
            components.iframe(look['embed_url'], height=720)

            st.markdown("## Query Result")
            query_result_df = looker_query_runner.run_query(write_query)
            st.table(query_result_df)
            
    except Exception as e:
        # print error and traceback
        print(f"error: {e}")
        print(traceback.format_exc())
        st.error(e)
