import re
from flask import Blueprint, request, render_template
from backend.config import Config
from qdrant_client import QdrantClient
import openai
import torch
import whisper
from datetime import datetime
from dotenv import load_dotenv

api = Blueprint(
    "api",
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
    static_url_path="/static",
)


@api.route("/", methods=["GET"])
def home():
    now = datetime.now()
    timestamp = str(now.hour) + ":" + str(now.minute)
    return render_template("chat.html", timestamp=timestamp)



import replicate
import os

def get_llama_response(prompt: str, max_tokens: int = 2048, stop: list = [], echo: bool = False) -> str:
    # Set your Replicate API token as an environment variable
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    
    # Define the input data for the Llama v2 model
    input_data = {
        "prompt": prompt,
        "system_prompt": "You are a compassionate medical chatbot here to provide support and accurate advice for health concerns. Your main goal is to offer helpful advice to users seeking assistance with their medical queries. If condition is urgent or severe, advise seeking immediate medical help. Remember, your name is ListenAI. Now, please anser my {question}:",
        "max_new_tokens": max_tokens,
        "temperature": 0.95,
        "top_p": 0.95,
        "top_k": 250,
        "repetition_penalty": 1.15,
        "repetition_penalty_sustain": 256,
        "token_repetition_penalty_decay": 128,
        "debug": False
    }

    # Define the model version ID
    model_version_id = "a16z-infra/llama-2-13b-chat:2a7f981751ec7fdf87b5b91ad4db53683a98082e9ff7bfd12c8cd5ea85980a52"

    # Run the model using the replicate.run function
    output = replicate.run(model_version_id, input=input_data)

    # Extract the response from the output
    response = "".join(output)

    return response




@api.route("/response")
def response():
    model = request.args.get("model")
    query = request.args.get("msg")
    questions = request.args.get("questions")
    answers = request.args.get("answers")

    questions = questions.split("|")[:-2]
    answers = answers.split("|")[:-1]

    messages = [
        {
            "role": "system",
            "content": "You are a compassionate medical chatbot here to provide support and accurate advice for health concerns. Your main goal is to offer helpful advice to users seeking assistance with their medical queries. If urgent or severe, advise seeking immediate medical help. Remember, your name is ListenAI. Now, please anser my {question}:",
        }
    ]
    for question, answer in zip(questions, answers):
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})


    for message in messages:
        print(message, flush=True)

    # Check if the selected model is "llama2" and get the response accordingly
    if model == "llama2":
        answer = get_llama_response(query)
    else:
        answer = get_response(query, messages)

    return answer


@api.route("/voice", methods=["POST"])
def voice():
    DEVICE = "cpu"

    f = request.files["audio_data"]
    with open("audio.wav", "wb") as audio:
        f.save(audio)

    model = whisper.load_model("base", device=DEVICE)
    result = model.transcribe("audio.wav")
    return result["text"]


def get_response(query: str, messages: list) -> str:
    openai.api_key = Config.OPENAI_KEY

    # connect to the cluster
    qdrant_client = QdrantClient(url=Config.CLUSTER_URL, api_key=Config.QDRANT_KEY)

    # create embedding for query
    response = openai.Embedding.create(input=query, model="text-embedding-ada-002")

    embeddings = response["data"][0]["embedding"]

    # search for similar embeddings
    search_result = qdrant_client.search(
        collection_name="my_collection", query_vector=embeddings, limit=5
    )

    # create prompt for GPT3.5
    prompt = "Context:\n"

    for result in search_result:
        prompt += result.payload["text"] + "\n---\n"
    prompt += "Question:" + query + "\n---\n" + "Answer:"

    messages.append({"role": "user", "content": prompt})
    print("sending answer to GPT ...", flush=True)

    # create answer
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    print("answer received!", flush=True)
    # get content
    answer = completion.choices[0].message.content

    return answer
