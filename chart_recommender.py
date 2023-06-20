# Class to generate queries to the Looker API using LangChain from natural language questions
import json
from langchain.chains import LLMChain
from langchain.base_language import BaseLanguageModel


class ChartRecommender:
    def prompt_template(self):
        prompt_template = """
        Given an input dataframe, first analyse the data inside.
        Then suggest a chart type that can best visualise the data.
        Only use the following chart types: {chart_types}
        Only respond in a valid JSON in the following format: {{"chart_type": "the chart type you recommend"}}
        If you don't know which chart type to reccomend just pick line chart to visualise instead of showing the original dataframe

        In looker, the chart types are mapped as:
        vertical bar chart as looker_column,
        horizontal bar chart as looker_bar,
        line chart as looker_line,
        area chart as looker_area,
        scattered chart as looker_scatter,
        single number chart as single_value,
        table is looker_grid

        Example:
        Given the following dataframe:
        date       value
        2023-05-01 20
        2023-05-02 21
        2023-05-03 22
        2023-05-05 23
        2023-05-06 24
        you should reply:
        {{
            "chart_type": "looker_line"
        }}

        Given the following dataframe:
        users.age                       order_items.first_purchase_count
        47                              1455
        48                              1454
        28                              1435
        53                              1403
        42                              1397
        59                              1396
        58                              1396
        36                              1391
        you should reply:
        {{
            "chart_type": "looker_bar"
        }}

        Given the following dataframe:
        order_facts.order_amount    orders.created_date
        4.500000                    2023-06-19
        25.000000                   2023-06-19
        56.000000                   2023-06-19
        49.990002                   2023-06-19
        7.500000                    2023-06-19
        8.000000                    2023-06-19
        9.500000                    2023-06-19
        you should reply:
        {{
            "chart_type": "looker_line"
        }}

        # Dataframe
        {dataframe}"""

        return prompt_template

    def __init__(self, llm: BaseLanguageModel):
        self.cr = LLMChain.from_string(llm=llm, template=self.prompt_template())

    def run(self, chart_types, dataframe):
        str_chart_types = ','.join(chart_types)
        response = self.cr.predict(chart_types=str_chart_types, dataframe=dataframe)
        print(f"raw response: {response}")
        response = response[response.find("{"): response.rfind("}") + 1]
        print(f"split response: {response}")

        return json.loads(response)
