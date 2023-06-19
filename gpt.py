import os
import openai
import pandas as pd

def read_api_key():
    api_key_file = os.path.expanduser("~/.apikey.secret")
    with open(api_key_file, "r") as file:
        return file.read().strip()

def read_csv(file_path):
    return pd.read_csv(file_path, delimiter=';')

def call_openai_api(row, instructions, model='text-davinci-003'):
    prompt = f"{instructions}\nCSV Data: {row}\n"
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=200,
        temperature=0.6
    )
    return response.choices[0].text.strip()

def generate_qa_pairs(df, instructions):
    for _, row in df.iterrows():
        row_str = '; '.join(str(item) for item in row)
        for _ in range(2):  # generate two questions per row
            output = call_openai_api(row_str, instructions)
            print(f"{output}\n")

def main():
    api_key = read_api_key()
    openai.api_key = api_key

    df = read_csv('input.csv')
    instructions = """
    This is what I expect from you:
    
    1. Access a CSV data that I will paste in the prompt. The CSV is comma-separated (;).
    
    2. The CSV Data contains various columns: 
    - Row: The row identifier. This value must not appear in the question nor the answer
    - Last-Update: The date of the last update.
    - RR Version: The version of the RR.
    - Identifier: The unique identifier.
    - Task: The description of the individual task.
    - Responsibility: Who is responsible of this task, if it's a standard service included in the contract or it's an additional service, etc...
    - Remarks: Additional remarks or notes for the task
    - CAS Package: Optional Cloud Architecture Services package that cover the execution of the task
    - Package Code: The code associated with the CAS package.
    - Delivery by: Who will deliver this task
    - Ordering Information: Information on how customers can request optional services.
    
    3. Use data from the Task, Responsibility, and Remarks columns to generate a series of Q&A pairs for a chatbot. Focus on the Task and Responsibility, but include all information from other columns in the answers.
    
    4. Each answer should include the data of the following columns: 
    - Identifier: [Identifier]
    - Task Group: [Task Group]
    - CAS Package: [CAS Package]
    - Package Code: [Package Code]
    - Delivery by: [Delivery by]
    - Ordering Information: [Ordering Information]
    - Last-Update: [Last-Update]
    - RR Version: [RR Version]
    
    5. Exclude the Task Group column in the questions, as it's internal information not known to customers.

        6. Generate a plain text output for each Q&A pair, consisting of two lines:
    - The first line should start with "Q:" and contain the question.
    - The second line should start with "A:" and contain the answer.

    7. Separate each Q&A pair by adding 2 blank lines.

    8. Replace any occurrence of "HEC" with "SAP ECS".

    9. Do not number the Q&A pairs; instead, label them with "Q:" and "A:".

    10. End every answer with the Last-Update and RR Version.

    11. Generate 2 questions per each row of the CSV data.
    """

    generate_qa_pairs(df, instructions)

if __name__ == "__main__":
    main()
