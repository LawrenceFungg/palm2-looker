# Class to generate queries to the Looker API using LangChain from natural language questions
import json
from webbrowser import Chrome
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.base_language import BaseLanguageModel

class QueryConverter:
    def prompt_template(self):
        prompt_template = """
        Given an input question, first create a syntactically correct JSON.
        Do not use "fields": ["*"] in the JSON. 
        Field names must include the view name. The JSON must include the view name.
        Must include the fields so that data is retrieved.
        Use only the explore name as the name of the view in the query, where the explore names are those after 'explore: '

        Example:
        Given a question: Give me a list of products with their ids and names
        you should reply:
        {{
            "view": "order_items",
            "fields": [
                "products.id",
                "products.name"
            ],
            "model": "thelook_bq"
        }}

        Given a question: Sum of total sale price per day over the previous 7 days
        {{
            "view": "order_items",
            "fields": [
                "order_items.total_sale_price",
                "orders.created_date"
            ],
            "filters": {{
                "orders.created_date": "last 7 days"
            }},
            "aggregations": [
                {{
                    "type": "sum",
                    "field": "order_items.total_sale_price"
                }}
            ],
            "model": "thelook_bq"
        }}

        # LookML Reference
        ```
        {context}
        ```

        # Question
        {question}"""

        return PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    def __init__(self, looker_model_name: str, docsearch: Chrome, llm: BaseLanguageModel):
        self.model_name = looker_model_name
        chain_type_kwargs = {"prompt": self.prompt_template()}
        self.qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)

    def run(self, question):
        response = self.qa.run(question)
        # extract the contents within {} from the response string
        print(f"raw response: {response}")
        response = response[response.find("{"): response.rfind("}") + 1]
        print(f"split response: {response}")

        response_json = json.loads(response)
        # Delete only model field from json
        if "model" in response_json:
            del response_json["model"]
        response_json["model"] = self.model_name

        # if "view" in response_json:
        #     del response_json["view"]
        # response_json["view"] = 'order_items'

        print(f"response_json: {response_json}")
        self.response_json = response_json

        return self.response_json
