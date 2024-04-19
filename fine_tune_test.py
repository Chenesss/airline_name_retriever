import pandas as pd
from openai import OpenAI
from openai_airline_name_extract import get_openai_response
import os

client = OpenAI()

def create_complete_conversation(row):
    system_content_dict = {"role": "system", "content": "Return a similarity value of 2 decimal places between 0  and 1. Only provide the number."}
    user_content_dict = {"role": "user", "content": row['prompt']}
    assistant_content_dict = {"role": "assistant", "content": row['completion']}

    return ({'messages': (system_content_dict, user_content_dict, assistant_content_dict)})



def create_jsonl(filename, output_file):
    df = pd.read_csv(filename, skiprows=1,names=['prompt', 'completion'])
    df['message'] = df.apply(create_complete_conversation, axis=1)

    with open(output_file,"w") as file_handle:
        data=df['message'].to_json(orient='records', lines=True)[0:-1] #df.to_json always has an extra line. Apparently its intended https://github.com/pandas-dev/pandas/issues/36888
        file_handle.write(data)

def upload_file(file_name, purpose) -> None:
    client.files.create(
        file=open(file_name, "rb"),
        purpose=purpose
    )


if __name__=='__main__':
    data_file="airline_train.csv"
    output_file = "airline_train.jsonl"

    #this is generates a JSONl file in the "conversational chat format" which works with gpt-3.5-turbo
    # You can read more about other example formats here https://platform.openai.com/docs/guides/fine-tuning/example-format
    create_jsonl(data_file, output_file)

