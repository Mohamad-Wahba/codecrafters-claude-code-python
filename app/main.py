import argparse
import os
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read",
                            }
                        },
                        "required": ["file_path"],
                    },
                },
            }
        ],
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr, end="")

    # TODO: Uncomment the following line to pass the first stage
    msg = chat.choices[0].message.content
    if msg is not None:
        print(msg, end="")

    # Tool Execution
    tool_calls = chat.choices[0].message.tool_calls
    if tool_calls is not None:
        for tool_call in tool_calls:
            if tool_call.function.name == "Read":
                args = json.loads(tool_call.function.arguments)
                file_path = args["file_path"]
                with open(file_path, "r") as f:
                    print(f.read(), end="")


if __name__ == "__main__":
    main()
