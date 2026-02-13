import os
import sys
import argparse

from openai import OpenAI


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    return args.p


def load_config():
    API_KEY = os.getenv(key="OPENROUTER_API_KEY")
    BASE_URL = os.getenv(
        key="OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1"
    )
    if not API_KEY:
        raise RuntimeError("API key not set")

    return API_KEY, BASE_URL


def build_client(api_key, base_url):
    client = OpenAI(api_key=api_key, base_url=base_url)

    return client


def run_prompt(client: OpenAI, prompt):
    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": prompt}],
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    return chat.choices[0].message.content


def main():
    prompt = parse_args()
    api_key, base_url = load_config()
    client = build_client(api_key, base_url)
    response = run_prompt(client, prompt)

    print("logs: ", file=sys.stderr)
    print(f"{response}")


if __name__ == "__main__":
    main()
