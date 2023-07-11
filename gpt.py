import os
import openai
import pandas as pd
import logging
import tiktoken
import argparse
import json
from typing import Tuple


class OpenAIAPI:
    """Handles communication with the OpenAI API."""

    def __init__(self, model: str):
        self.model = model
        self.api_key = self._load_api_key()
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _load_api_key(self) -> str:
        try:
            api_key_file = os.path.expanduser("~/.apikey.secret")
            with open(api_key_file, "r") as file:
                return file.read().strip()
        except Exception as e:
            logging.error("Failed to read API key: %s", e)
            raise

    def call(self, row: str, instructions: str) -> str:
        prompt = f"{instructions}\nCSV Data: {row}\n"
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=200,
            temperature=0.6
        )
        return response.choices[0].text.strip()

    def get_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))


class CSVReader:
    """Loads and processes CSV data."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_csv()

    def _load_csv(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.file_path, delimiter=';')
        except Exception as e:
            logging.error("Failed to read CSV file: %s", e)
            raise


class Prompt:
    """Loads and processes the prompt."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text, self.tokens = self._load_prompt()

    def _load_prompt(self) -> Tuple[str, int]:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            with open(self.file_path, "r") as file:
                prompt = file.read()
                return prompt, len(encoding.encode(prompt))
        except Exception as e:
            logging.error("Failed to read prompt: %s", e)
            raise


def generate_qa_pairs(df: pd.DataFrame, instructions: str, format: str, api: OpenAIAPI) -> int:
    """Generate Q&A pairs and save to file in specified format."""
    total_tokens = 0
    qa_pairs = []
    for _, row in df.iterrows():
        row_str = '; '.join(str(item) for item in row)
        output = api.call(row_str, instructions)
        qa_pairs.append(output)
        total_tokens += api.get_tokens(row_str + instructions + output)
        print(output)

    if format.lower() == 'json':
        with open('output.json', 'w') as f:
            json.dump(qa_pairs, f)
    else:  # default to TXT
        with open('output.txt', 'w') as f:
            for pair in qa_pairs:
                f.write("%s\n" % pair)

    return total_tokens



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", help="Output format (TXT or JSON), default is TXT", default='txt')
    parser.add_argument("--model", help="The model to be used, default is text-davinci-003", default='text-davinci-003')
    return parser.parse_args()


def print_initialization(api_key: str, model: str, format: str, df: pd.DataFrame, prompt: str, prompt_tokens: int) -> None:
    """Print initialization progress."""
    print("Initialization:")
    print(f"API key loaded: {'Yes' if api_key else 'No'}")
    print(f"Model selected: {model}")
    print(f"Output format selected: {format.upper()}")
    print(f"CSV file loaded: {'Yes' if not df.empty else 'No'}")
    print(f"Prompt loaded: {'Yes' if prompt else 'No'}")
    print(f"Prompt tokens: {prompt_tokens}")


def main() -> None:
    args = parse_args()

    api = OpenAIAPI(args.model)
    openai.api_key = api.api_key

    reader = CSVReader('input.csv')
    prompt = Prompt('prompt.txt')

    print_initialization(api.api_key, args.model, args.format, reader.data, prompt.text, prompt.tokens)

    total_tokens = generate_qa_pairs(reader.data, prompt.text, args.format, api)
    print(f"Total tokens used: {total_tokens}")


if __name__ == "__main__":
    main()
