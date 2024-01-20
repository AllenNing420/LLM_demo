# -*- coding: utf-8 -*-
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

api_key = "sk-j3e4GaE8bIiHEEbJxlo3YoECiEXLYIu2THwrNrkOIDUcSEQx"
# api_key = "sk-4C0P5RQGyRrxak3X7XTlcyanDs4JGjrxhtn3dRUhoIBSPw4J"

def langchain_openai_demo():
    prompt = ChatPromptTemplate.from_template("{topic}需要注意什么？")
    model = ChatOpenAI(openai_api_key=api_key,
                       base_url="https://api.chatanywhere.tech/v1",
                       model="gpt-3.5-turbo")
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    print(chain.invoke({"topic": "跑步"}))


def openai_demo():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=api_key,
        base_url="https://api.chatanywhere.tech/v1"
    )
    message = [{'role': 'user', 'content': '鲁迅和周树人的关系'}, ]
    gpt_35_api(client, message)
    gpt_35_api_stream(client, message)


# 非流式
def gpt_35_api(client, messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    print(completion.choices[0].message.content)


# 流式
def gpt_35_api_stream(client, messages: list):
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息
    """
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


if __name__ == '__main__':
    langchain_openai_demo()
    openai_demo()
