import json
import os

from google import genai
from google.genai import types
from pydantic import BaseModel

client = genai.Client()


class Question(BaseModel):
    question: str
    # options: list[str]
    # answer: str
    A: str
    B: str
    C: str
    D: str
    answer: str


for video_file in os.listdir("tmp"):
    if not video_file.endswith(".mp4"):
        continue
    print("--" * 40)
    print(f"Processing {video_file}")
    with open(f"tmp/{video_file}", "rb") as f:
        video_bytes = f.read()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=video_bytes,
                mime_type="video/mp4",
            ),
            """This is a video of assembly line workers. 
        Please focus on the hand actions of the workers and ask three single-choice questions related to the *hand spatial relationship* and *hand information*. 
        Try to figure out which hand performs the action.
        Answers should be included and responses should be in JSON format. For example,

        {
            "question":"What is the man doing with his hands?",
            "A": "Both hands are not visible and not interacting with any objects.",
            "B": "Holding an object in his right hand.",
            "C": "Holding an object in his left hand.",
            "D": "Using one hand to gesture.",
            "answer": "C"
        }        
        """,
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Question],
        },
    )
    print(response.text)
    output = json.loads(response.text)

    with open(f"tmp/{video_file}.json", "w") as f:
        json.dump(
            {
                "human": {
                    "questions": output,
                }
            },
            f,
            indent=2,
        )
