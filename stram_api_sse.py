import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette import ServerSentEvent, EventSourceResponse
from transformers import AutoTokenizer, AutoModel

app = FastAPI()


class ChatBody(BaseModel):
    question: str


def gen_chat(chat: ChatBody = None, his=None):
    with torch.autocast("cuda"):
        start_idx = 0
        end_idx = 0
        for response, history in model.stream_chat(tokenizer, chat.question, his):
            end_idx = len(response)
            this_response = response[start_idx:end_idx]
            start_idx = end_idx
            yield {"response": this_response, "finished": False}
        yield {"finished": True}


@app.post("/chat_stream")
async def chat_stream(chat_body: ChatBody = None):
    '''
     request json format : {"question":"介绍一下你自己吧"}
    :param chat_body:
    :return:
    '''

    def decorate(generator):
        for item in generator:
            yield ServerSentEvent(item, event='answer')

    return EventSourceResponse(decorate(gen_chat(chat_body)))


if __name__ == "__main__":
    global tokenizer, model
    MODEL_PATH = '/data/model/chatglm3-6b'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).quantize(8).cuda()
    uvicorn.run(app, host="0.0.0.0", port=6689)
