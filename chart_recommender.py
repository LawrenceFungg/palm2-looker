# Class to generate queries to the Looker API using LangChain from natural language questions
import json
from langchain.chains import LLMChain
from langchain.base_language import BaseLanguageModel


class ChartRecommender:
    def prompt_template_chart_type(self):
        # prompt_template = """
        # Analyse the data inside
        # Then suggest a chart type that can best visualise the data.
        # Only use the following chart types: {chart_types}
        # Only respond in a valid JSON in the following format: {{"chart_type": "the chart type you recommend"}}
        # If you don't know which chart type to reccomend just pick line chart to visualise instead of showing the original dataframe

        # In looker, the chart types are mapped as:
        # vertical bar chart as looker_column,
        # horizontal bar chart as looker_bar,
        # line chart as looker_line,
        # area chart as looker_area,
        # scattered chart as looker_scatter,
        # single number chart as single_value,
        # table is looker_grid

        # Example:
        # Given the following dataframe:
        # date       value
        # 2023-05-01 20
        # 2023-05-02 21
        # 2023-05-03 22
        # 2023-05-05 23
        # 2023-05-06 24
        # you should reply:
        # {{
        #     "chart_type": "looker_line"
        # }}

        # Given the following dataframe:
        # users.age                       order_items.first_purchase_count
        # 47                              1455
        # 48                              1454
        # 28                              1435
        # 53                              1403
        # 42                              1397
        # 59                              1396
        # 58                              1396
        # 36                              1391
        # you should reply:
        # {{
        #     "chart_type": "looker_bar"
        # }}

        # Given the following dataframe:
        # orders.created_date  order_items.total_sale_price
        # 2023-07-03	         54815.090047836304	
        # 2023-07-02	         41824.160017967224	
        # 2023-07-01	         140062.61022400856	
        # 2023-06-30	         90330.29001188278	
        # 2023-06-29	         66915.59010267258	
        # 2023-06-28	         46527.80003905296
        # 2023-06-27           32893.730028
        # you should reply:
        # {{
        #     "chart_type": "looker_line"
        # }}

        # # Dataframe
        # {dataframe}"""

        prompt_template = """
            You are a data expert and based on the data input, please suggest a chart type that best represent the data.
            Only use the following chart types: {chart_types}

            Example:
            Given the following dataframe:
            products.id                                    products.name     products.brand
            1     Seven7 Women's Long Sleeve Stripe Belted Top             Seven7
            2   Calvin Klein Women's MSY Crew Neck Roll Sleeve       Calvin Klein
            3   Calvin Klein Jeans Women's Solid Flyaway Shirt Calvin Klein Jeans
            4                   Bailey 44 Women's Undertow Top          Bailey 44
            5 Anne Klein Women's Plus-Size Button Front Blouse         Anne Klein
            6   Wilt Women's Color Blocked Big Mixed Slant Top               Wilt
            7                     Lucky Brand Women's Riad Tee        Lucky Brand
            8         Ella Moss Women's Stella Button Up Shirt          Ella Moss
            9   Alternative Women's Alice Drop Shoulder V-Neck        Alternative
            10   Calvin Klein Women's Plus-Size Print Drape Top       Calvin Klein
            you should suggest looker_grid
            
            Data: {dataframe}
        """
        return prompt_template
    
    def prompt_template_json_format(self):
        prompt_template = """
            Given the following chart recommendation, extract the chart type recommended and turn it into a JSON with the following mapping:

            For example, for a line chart, respond {{"chart_type": "looker_line"}}

            Chart recommendation: {chart_rec}
        """
        return prompt_template

    def __init__(self, llm: BaseLanguageModel):
        self.cr_chart_type = LLMChain.from_string(llm=llm, template=self.prompt_template_chart_type())
        self.cr_json_format = LLMChain.from_string(llm=llm, template=self.prompt_template_json_format())

    def run(self, chart_types, dataframe):
        str_chart_types = ','.join(chart_types)
        response_chart_type = self.cr_chart_type.predict(chart_types=str_chart_types, dataframe=dataframe)
        print(f"raw response: {response_chart_type}")

        response_chart_json = self.cr_json_format.predict(chart_rec=response_chart_type)
        response_chart_json = response_chart_json[response_chart_json.find("{"): response_chart_json.rfind("}") + 1]
        print(f"split response: {response_chart_json}")

        return json.loads(response_chart_json)
