from openai import OpenAI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import os


client = OpenAI()
MODEL_NAME="gpt-3.5-turbo" #### You can change the model here and use your own fine tuned models
TEMPERATURE=0

def get_openai_response(data_df, prompt_column, response_column, system_content="", model=MODEL_NAME):
    for index, row in data_df.iterrows():
        response = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": row[prompt_column]}
            ]
        )
        
        data_df.loc[index, response_column] = response.choices[0].message.content
    
    print(data_df)
    return data_df

def show_bar_chart(df, similarity_rating_column):
    def get_similarity_group(similiarty):
        similiarty=float(similiarty)
        if similiarty == 1:
            return "1"
        elif similiarty >= 0.6 and similiarty < 1:
            return "0.60-0.99"
        else:
            return "<0.6"

    df['similarity_score'] = df[similarity_rating_column].apply(get_similarity_group)


    df_per_airline = df[['tweet', 'similarity_score']].groupby(["similarity_score"]).count()
    print(df_per_airline) 
    df_per_airline['percent'] = df_per_airline.div(df_per_airline.sum()).mul(100).round(1)

    print(df_per_airline)
    ax=df_per_airline.plot(kind='bar', y='tweet')
    labels=df_per_airline['percent'].astype('str') + '%'
    # labels=df_per_airline['tweet']

    for container in ax.containers:
        ax.bar_label(container, labels=labels)

    plt.show()


if __name__=='__main__':
    # Read test data into dataframe
    df = pd.read_csv("airline_test.csv")

    # Generate the prompts to get airline from tweet and get openai response
    df['user_prompt'] = df['tweet'].apply(lambda x: f"Can you tell me what airlines are mentioned in this tweet? '{x}'")
    df = get_openai_response(data_df=df, prompt_column='user_prompt',response_column="openai_response", system_content="You are a helpful assistant that will respond in this format \"['airline 1', 'airline 2', 'airline 3']\"")

    # Generate the prompts to ask OpenAI how similar does it think the zeroprompt responses are vs the pre-determined answers
    # Always using gpt-3.5-turbo because Fine Tuned models may give wrong answers depending on the tuning
    df['user_prompt_similarities'] = df.apply(lambda x: f"How similar are the sets '{x['airlines']}' and '{x['openai_response']}'", axis=1)
    df = get_openai_response(data_df=df, prompt_column='user_prompt_similarities',response_column="response_similarity_rating", system_content="Return a similarity value of 2 decimal places between 0  and 1. Only provide the number.", model='gpt-3.5-turbo')

    df.to_csv(f"Results.csv", index=False)

    show_bar_chart(df, 'response_similarity_rating')
