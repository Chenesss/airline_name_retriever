# airline_name_retriever
Using OpenAI's model, retrieve Airline names from any text source

You will need to run 'openai_airline_name_extract.py' first which will generate a Results.csv file.
'openai_embedding_airline_name.py' will rely on the Results.csv for some data in its logic
There is a global variable in the script for the model used. I have set it to  "gpt-3.5-turbo" by default but you can change this. This will come in handy when you are looking into Fine Tuning models.
I have set the temperature variable to 0. This ensures that we get consistent results without deviation. 
