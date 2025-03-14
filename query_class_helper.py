from pydantic import BaseModel
from jinja2 import Template
import json

import sys
sys.stdout.reconfigure(encoding='utf-8')  # Force UTF-8 encoding

Template_str_Context = """
[
    {"role": "system", "content": "You are a bot that specializes in data augmentation and outputs JSON format. You will receive two contexts and generate a new context under the key 'context' that is related to the same subject (don't output the other context.). The context must be short and can be either a question or a description."},
    {"role": "user", "content": {{ ("context 1: " ~ context1 ~ "\ncontext 2: " ~ context2 ~ ("\nPrevious Error of your generation: " ~ error if error else "None") ~ ("\nPrevious Answer of your generation change it accordingly: " ~ previous_answer if previous_answer else "None")) | tojson }} }
]
"""

Template_str_QA = """[
    {"role": "system", "content": "You are a bot that specializes in data augmentation and outputs JSON format. You will generate a new very exhaustive question-answer pair under the keys 'Question' and 'Answer' based on the given context. Take inspiration from the Stack Overflow style of conversation (that will be provided in the example), create a little story or explain how you got to the problem before asking the question, and don't output anything else than the JSON itself (what's inside the key must be just a string). Two examples will be given by the user. Use tags like in the examples in the question-answer pair generated."},
    {"role": "user", "content": {{ ("Example 1:\ncontext: " ~ context1 ~ "\nquestion: " ~ question1 ~ "\nanswer: " ~ answer1 ~ "\nExample 2:\ncontext: " ~ context2 ~ "\nquestion: " ~ question2 ~ "\nanswer: " ~ answer2 ~ "\n\nNow generate based on the next context:\ncontext: " ~ context3 ~ ("\nPrevious Error of your generation: " ~ error if error else "None") ~ ("\nPrevious Answer of your generation change it accordingly: " ~ previous_answer if previous_answer else "None") ) | tojson }} }
]"""

Template_str_A = """[
    {"role": "system", "content": "You are a bot that specializes in data augmentation and outputs JSON format. You will generate a new very exhaustive answer from a given question under the key 'Answer' based on the given context. Take inspiration from the Stack Overflow style of conversation (that will be provided in the example) and don't output anything else than the JSON itself (what's inside the key must be just the answer in a string format). Two examples will be given by the user."},
    {"role": "user", "content": {{ ("Example 1:\ncontext: " ~ context1 ~ "\nquestion: " ~ question1 ~ "\nanswer: " ~ answer1 ~ "\nExample 2:\ncontext: " ~ context2 ~ "\nquestion: " ~ question2 ~ "\nanswer: " ~ answer2 ~ "\n\nNow generate based on the next context:\ncontext: " ~ context3 ~ " \nquestion: " ~ question3 ~ ("\nPrevious Error of your generation: " ~ error if error else "None") ~ ("\nPrevious Answer of your generation change it accordingly: " ~ previous_answer if previous_answer else "None")) | tojson }} }
]"""

Template_str_consistency = """
[
    {"role": "system", "content": "You are an AI specializing in answer consistency checking. Your task is to determine whether two responses to the same question are coherent and whether they align with the question itself. You will output JSON format with a \"consistency\" field (\"Yes\" or \"No\") and a \"score\" field (between 0 and 1)."},
    {"role": "user", "content": {{ (
        "Question: " + question + "\\n" +
        "1st answer: " + answer1 + "\\n" +
        "2nd answer: " + answer2 + "\\n" +
        "Analyze whether both answers provide the same core information and whether they are relevant and consistent with the question. Use a chain-of-thought approach:\\n" +
        "1. Identify the key elements in the question.\\n" +
        "2. Extract the main points from each answer.\\n" +
        "3. Compare the answers against each other and the question.\\n" +
        "4. Determine if both answers are saying the same thing and if they correctly address the question.\\n" +
        "5. Assign a consistency score based on the level of agreement.\\n" +
        "Output JSON in the format: {\\\"reasoning\\\": \\\"<your explanation>\\\", \\\"consistency\\\": \\\"<Yes/No>\\\", \\\"score\\\": <value between 0 and 1>}"
    ) | tojson }}
]
"""


class query:
    def __init__(self):
        self.context1 = None
        self.context2 = None
        self.context3 = None
        self.question1 = None
        self.question2 = None
        self.answer1 = None
        self.answer2 = None
    
    def create_context(self,context1:str,context2:str,Error="None",previous_answer="None") -> dict:
        context1.replace('"', '\\"')
        context2.replace('"', '\\"')
        Error.replace('"', '\\"')
        previous_answer.replace('"', '\\"')
        data={"context1":context1,
              "context2":context2,
              "error":Error,
              "previous_answer":previous_answer}
        template = Template(Template_str_Context)
        messages = json.loads(template.render(**data))
        self.context1 = context1,
        self.context2 = context2
        return messages
    
    def create_QA(self,question1,question2,answer1,answer2,context3,Error="None",previous_answer="None") -> dict:
        question1.replace('"', '\\"')
        question2.replace('"', '\\"')
        answer1.replace('"', '\\"')
        answer2.replace('"', '\\"')
        context3.replace('"', '\\"')
        Error.replace('"', '\\"')
        previous_answer.replace('"', '\\"')
        data = {"context1":self.context1,
                "context2":self.context2,
                "context3":context3,
                "question1":question1,
                "question2":question2,
                "answer1":answer1,
                "answer2":answer2,
                "error":Error,
                "previous_answer":previous_answer}
        template = Template(Template_str_QA)
        messages = json.loads(template.render(**data))
        self.context3 = context3
        self.question1=question1
        self.question2=question2
        self.answer1=answer1
        self.answer2=answer2
        return messages
        
    def create_A(self,question3,Error="None",previous_answer="None") -> dict:
        question3.replace('"', '\\"')
        Error.replace('"', '\\"')
        previous_answer.replace('"', '\\"')
        data = {"context1":self.context1,
                "context2":self.context2,
                "context3":self.context3,
                "question1":self.question1,
                "question2":self.question2,
                "answer1":self.answer1,
                "answer2":self.answer2,
                "question3":question3,
                "error":Error,
                "previous_answer":previous_answer}
        template = Template(Template_str_A)
        messages = json.loads(template.render(**data))
        return messages
    
    # def Get_consistency(self,Answer1,Answer2,Question) -> dict:
    #     Answer1.replace('"', '\\"')
    #     Answer2.replace('"', '\\"')
    #     Question.replace('"', '\\"')
    #     data = {"answer1":Answer1,"answer2":Answer2,"question":Question}
    #     template = Template(Template_str_consistency)
    #     messages = json.loads(template.render(**data))
    #     return messages
    
    def Get_consistency(self, Answer1, Answer2, Question) -> list:
        system_content = "You are an AI specializing in answer consistency checking. Your task is to determine whether two responses to the same question are coherent and whether they align with the question itself. You will output JSON format with a \"consistency\" field (\"Yes\" or \"No\") and a \"score\" field (between 0 and 1)."
        
        user_content = (
            f"Question: {Question}\n"
            f"1st answer: {Answer1}\n"
            f"2nd answer: {Answer2}\n"
            "Analyze whether both answers provide the same core information and whether they are relevant and consistent with the question. Use a chain-of-thought approach:\n"
            "1. Identify the key elements in the question.\n"
            "2. Extract the main points from each answer.\n"
            "3. Compare the answers against each other and the question.\n"
            "4. Determine if both answers are saying the same thing and if they correctly address the question.\n"
            "5. Assign a consistency score based on the level of agreement.\n"
            "Output JSON in the format: {\"reasoning\": \"<your explanation>\", \"consistency\": \"<Yes/No>\", \"score\": <value between 0 and 1>}"
        )
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        return messages