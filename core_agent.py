import boto3
import time
import openai
import google.generativeai as genai

class CoreAgent:
    def __init__(self, system_prompt, model_id='anthropic.claude-3-haiku-20240307-v1:0', region_name='us-west-2'):
        self.model_id = model_id
        self.system_prompt = system_prompt
        if 'gpt' in self.model_id:
            self.openai_client = openai.OpenAI()
        elif 'gemini' in self.model_id:
            self.gemini_client = genai.GenerativeModel(
                model_name=self.model_id,
                system_instruction=self.system_prompt
            )
        else:
            self.brt = boto3.client(service_name='bedrock-runtime', region_name=region_name)

    def query(self, input_str):
        user_message = {"role": "user", "content": [{"text": input_str}]}
        messages = [user_message]
        FAILURE_COUNTER = 0

        while True:
            try:
                if 'gpt' in self.model_id:
                    completion = self.openai_client.chat.completions.create(
                        model=self.model_id,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {
                                "role": "user",
                                "content": input_str
                            }
                        ]
                    )
                    llm_response = completion.choices[0].message.content
                elif 'gemini' in self.model_id:
                    # messages = transform_to_gemini([
                    #         {"role": "system", "content": self.system_prompt},
                    #         {
                    #             "role": "user",
                    #             "content": input_str
                    #         }
                    #     ])
                    # print(messages)
                    messages = [input_str]
                    response = self.gemini_client.generate_content(messages)
                    llm_response = response._result.candidates[0].content.parts[0].text
                else:
                    response = self.brt.converse(
                        modelId=self.model_id,
                        messages=messages,
                        system=[{"text": self.system_prompt}],
                    )
                    llm_response = response['output']['message']['content'][0]['text']

                break

            except Exception as e:
                FAILURE_COUNTER += 1
                if FAILURE_COUNTER >= 5:
                    return "Error: Failed to get a response after multiple attempts."
                print(f'Exception encountered: {e}. Retrying in 60 seconds...')
                time.sleep(60)

        # llm_response = response['output']['message']['content'][0]['text']
        return llm_response