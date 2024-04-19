import pandas as pd
import numpy as np
from openai import OpenAI
from ast import literal_eval
import os
from sklearn.metrics.pairwise import cosine_similarity
from openai_airline_name_extract import get_openai_response, show_bar_chart

client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def create_embedding_file(input_filename, column_name, output_file_name):
    '''
    Using a dictionary to track distinct airlines and get the embeddings for them
    Skip Airlines which we have already gotten embeddings for
    '''
    df = pd.read_csv(input_filename)

    airline_dict = {}
    for index, row in df.iterrows():
        for airline in literal_eval(row[column_name]): #Because there could be multiple airlines per tweet, we need to loop through the list
            try:
                value = airline_dict[airline]
            except KeyError:
                print(f'Getting embedding for text: {airline}')
                airline_dict[airline] = get_embedding(airline)
            except:
                print('Unexpected error')
                raise
            else:
                next

    airline_df = pd.DataFrame(airline_dict.items()).rename(columns={0:column_name, 1:'embeddings'})

    # write embeddings to file so we dont need to call it again
    airline_df.to_csv(output_file_name, index=False)


def calculate_similarity_score(target_df, search_df, similarity_function):
    '''
    Add an extra column to the dataset with the cosine_similarities values
    Return a dataframe
    '''
    for index, row in search_df.iterrows():
        target_df['cosine_similarities'] = target_df['embeddings'].apply(lambda x: similarity_function([literal_eval(x)], [literal_eval(row['embeddings'])]))
        temp_df=target_df.sort_values(by='cosine_similarities', ascending=False).head(1)

        # print(temp_df['airlines'].values[0])
        search_df.loc[index, 'embedding_assisted_response'] = temp_df['airlines'].values[0]
        search_df.loc[index, 'cosine_similarities'] = temp_df['cosine_similarities'].values[0]

    print(search_df)
    return search_df

def find_closest_name(data_df, data_column, train_embedding_df, searchterm_embedding_df, similarity_function, confidence_threshold):
    '''
    Using the cosine_similarities values and above a certain confidence_threshold, we will use the closest name 'verified' name of an Airline. (Example: 'Southwest Air' becomes 'Southwest Airlines')
    '''
    # converting it to a dictionary as we know the keys are unique so its easier to use
    similarity_score_df = calculate_similarity_score(target_df=train_embedding_df, search_df=searchterm_embedding_df, similarity_function=similarity_function)
    similarity_score_df.set_index(data_column,inplace=True)
    similarity_score_dict = similarity_score_df.to_dict(orient='index')

    # loop through data_df which has all the current responses
    for index, row in data_df.iterrows():
        temp_list = []
        for openai_response_airline in literal_eval(row[data_column]):
            closest_name, similarity_rating = similarity_score_dict[openai_response_airline]['embedding_assisted_response'], similarity_score_dict[openai_response_airline]['cosine_similarities']
            if similarity_rating > confidence_threshold: #Only swap the Airline if its above 0.9 in the similarity rating
                temp_list.append(closest_name)
            else:
                temp_list.append(openai_response_airline)

        data_df.loc[index, 'embedding_assisted_response'] = str(temp_list)
        
    # print(data_df)
    return data_df



if __name__ == '__main__':

    #Pull embedding data using the airline_train.csv file. Writing to file so that we don't have to regenerate it everytime
    if not os.path.isfile("airline_train_embedding.csv"):
        print("Unable to find airline_train_embedding.csv. Recalculating training set embedding.")
        create_embedding_file(input_filename='airline_train.csv', column_name='airlines', output_file_name="airline_train_embedding.csv")

    #Pull embedding data using the results file we generated from openai_airline_name_extract.py. Writing to file so that we don't have to regenerate it everytime
    results_file = 'Results.csv'
    data_df = pd.read_csv(results_file)
    if not os.path.isfile("airline_openai_response_embedding.csv"):
        print("Unable to find airline_train_eairline_openai_response_embeddingmbedding.csv. Recalculating zeroprompt response set embedding.")
        create_embedding_file(input_filename=results_file, column_name='openai_response', output_file_name="airline_openai_response_embedding.csv")


    train_embedding_df = pd.read_csv("airline_train_embedding.csv")
    searchterm_embedding_df = pd.read_csv("airline_openai_response_embedding.csv")

    result_df = find_closest_name(data_df=data_df, data_column='openai_response',train_embedding_df=train_embedding_df, searchterm_embedding_df=searchterm_embedding_df, similarity_function=cosine_similarity, confidence_threshold=0.9)

    # Generate the prompts to ask OpenAI how similar does it think the zeroprompt responses are vs the pre-determined answers
    result_df['user_prompt_embedding_result_similarities'] = result_df.apply(lambda x: f"How similar are the sets '{x['airlines']}' and '{x['embedding_assisted_response']}'", axis=1)
    result_df = get_openai_response(data_df=result_df, prompt_column='user_prompt_embedding_result_similarities',response_column="embedding_similarity_rating", system_content="Return a similarity value of 2 decimal places between 0  and 1. Only provide the number.", model='gpt-3.5-turbo')

    result_df.to_csv('Results.csv')

    show_bar_chart(result_df, 'embedding_similarity_rating')