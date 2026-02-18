import argparse
import os
import json
import subprocess
import shlex

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]

    while True:
        chat: ChatCompletion = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Write",
                        "description": "Write content to a file",
                        "parameters": {
                            "type": "object",
                            "required": ["file_path", "content"],
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The path of the file to write to",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content to write to the file",
                                },
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Execute a shell command",
                        "parameters": {
                            "type": "object",
                            "required": ["command"],
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The command to execute",
                                }
                            },
                        },
                    },
                },
            ],
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in result")

        msg: ChatCompletionMessage = chat.choices[0].message

        messages.append(
            {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": [tc.to_dict() for tc in msg.tool_calls]
                if msg.tool_calls
                else None,
            }
        )

        # Tool Execution
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "Read":
                    args = json.loads(tool_call.function.arguments)
                    file_path = args["file_path"]
                    with open(file_path, "r") as f:
                        content = f.read()

                if tool_call.function.name == "Write":
                    args = json.loads(tool_call.function.arguments)
                    file_path = args["file_path"]
                    content = args["content"]
                    with open(file_path, "w") as f:
                        f.write(content)

                if tool_call.function.name == "Bash":
                    args = json.loads(tool_call.function.arguments)
                    command = args["command"]

                    #! test
                    # print(f"-- args: {args} \n-- command:{command}")

                    command_run = subprocess.run(
                        args=shlex.split(command), capture_output=True, text=True
                    )
                    if not command_run.returncode:
                        content = command_run.stdout
                    else:
                        content = command_run.stderr

                #! test
                # print(f"-- content: {content}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": content,
                    }
                )

        if chat.choices[0].finish_reason == "stop":
            print(msg.content, end="")
            break


if __name__ == "__main__":
    main()
