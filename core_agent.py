import anthropic
import os
import time
from dotenv import load_dotenv
from state import *


load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

class CoreAgent():
    def __init__(self, model_id):
        self.model_id = model_id

    def query(self, input_str, system_prompt):
        user_message = {"role": "user", "content": input_str}
        FAILURE_COUNTER = 0

        while True:
            try: 
                response = client.messages.create(
                    model=self.model_id,
                    max_tokens=1000,
                    temperature=1,
                    system=system_prompt,
                    messages=[user_message]
                )
                return response
            
            except Exception as e:
                FAILURE_COUNTER += 1
                if FAILURE_COUNTER >= 5:
                    return "Error: Failed to get a response after multiple attempts."
                print(f'Exception encountered: {e}. Retrying in 60 seconds...')
                time.sleep(60)

    def query_tools(self, input_str, tools, system_prompt):
        user_message = {"role": "user", "content": input_str}
        FAILURE_COUNTER = 0

        while True:
            try: 
                response = client.messages.create(
                    model=self.model_id,
                    max_tokens=1000,
                    temperature=1,
                    tools=tools,
                    system=system_prompt,
                    messages=[user_message]
                )
                return response
            
            except Exception as e:
                FAILURE_COUNTER += 1
                if FAILURE_COUNTER >= 5:
                    return "Error: Failed to get a response after multiple attempts."
                print(f'Exception encountered: {e}. Retrying in 60 seconds...')
                time.sleep(60)


# Example usage
if __name__ == "__main__":
    agent = CoreAgent(model_id="claude-sonnet-4-20250514")

    # Normal LLM Query
    response = agent.query(system_prompt="You are a helpful chatbot that should answer inquiries politely and sincerely.", 
                           input_str="List all the python files in the current directory.")
    print(response.content[0].text)


    # Query with Bash tool
    tools=[
        {
            "type": "bash_20250124",
            "name": "bash"
        }
    ]
    response = agent.query_tools("List all the python files in the current directory.", tools=tools)
    for content in response.content:
        if content.type == "tool_use" and content.name == "bash":
            command = content.input.get("command")
        if content.type == 'text':
            text = content.text
    action = Action(command=command, description=text)
    print(action)

    