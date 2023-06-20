import pandas as pd
from chart_recommender import ChartRecommender
from langchain.llms import VertexAI
from shortuuid import uuid

class LookerLookCreator:
    def __init__(self, sdk) -> None:
        self.sdk = sdk

    def create_look(self, query, chart_types):
        print(type(query))
        print(query)
        
        # Initialize the chart recommender
        llm = VertexAI()
        cr = ChartRecommender(llm=llm)

        # Get the recommended chart type from LLM
        query_result = self.sdk.run_inline_query("json", query)
        query_result_df = pd.read_json(query_result)
        query_result_df = query_result_df.head(10)
        print(query_result_df.to_string(index=False))
        rec_chart_type = cr.run(chart_types=chart_types, dataframe=query_result_df.to_string(index=False))['chart_type']
        print(f'The recommended chart type is: {rec_chart_type}')

        # Inject the chart type into the query and add an artificial limit ot safeguard the look
        query['vis_config'] = {}
        query['vis_config']['type'] = rec_chart_type
        if 'limit' in query:
            if query['limit'] > 200:
                query['limit'] = 200
        else:
            query['limit'] = 200

        # Create and save the query for the look
        query_create_result = self.sdk.create_query(query)
        query_id = query_create_result['id']
        print(f'Query created with id {query_id}')

        # Create the look
        create_look_req = {
            'title': f'Visualised by Palm2 - {uuid()[:8]}',
            'user_id': '8',
            'folder_id': '6',
            'public': True,
            'query_id': f"{query_id}",
            'description': 'The chart type is recommended by Palm2 by looking at the data in the query'
        }
        create_look_result = self.sdk.create_look(create_look_req)

        return create_look_result
