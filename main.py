from llama_cpp import Llama
import pandas as pd
from query_class_helper import query
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

MODEL_PATH = "D:\Code Perso\Python\model\Llama-3.2-3B-Instruct-Q8_0.gguf"
TAG = "vector"
NUMBER_TO_GEN = 1
BAILOUT = 3

def sentence_similarity(sent1, sent2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sent1, sent2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity * 100, 2)  # Convert to percentage
def extract_json_objects(text: str):
    """
    Detects and extracts one or multiple JSON objects from a given string.

    Args:
        text (str): The input string potentially containing JSON objects.

    Returns:
        list: A list of parsed JSON objects (dictionaries).
    """
    json_objects = []
    brace_stack = []
    json_candidates = []

    start = None
    for i, char in enumerate(text):
        if char == "{":
            if start is None:
                start = i
            brace_stack.append(i)
        elif char == "}":
            if brace_stack:
                brace_stack.pop()
                if not brace_stack:
                    json_candidates.append(text[start : i + 1])
                    start = None

    for candidate in json_candidates:
        try:
            json_obj = json.loads(candidate)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    return json_objects
def generate_context(random_rows):
    counter_bailout_Context = 0
    while counter_bailout_Context < BAILOUT:
        context = llm.create_chat_completion(
            messages=query_helper.create_context(
                random_rows.iloc[0]["Title"], random_rows.iloc[1]["Title"]
            ),
            max_tokens=4096,
            temperature=0.5,
            # top_k=40,
            # top_p=0.80,
            # grammar="json"
        )["choices"][0]["message"]["content"]
        json_objects = extract_json_objects(context)
        for object in json_objects:
            if "context" in object:
                context3 = object["context"]
                return context3
        counter_bailout_Context +=1
    return None
def generate_QA(random_rows,context3):
    counter_bailout_QA = 0
    while counter_bailout_QA < BAILOUT:
        q_a = llm.create_chat_completion(
            messages=query_helper.create_QA(
                random_rows.iloc[0]["QuestionBody"],
                random_rows.iloc[1]["QuestionBody"],
                random_rows.iloc[0]["AnswerBody"],
                random_rows.iloc[1]["AnswerBody"],
                context3,
            ),
            max_tokens=4096,
            temperature=0.3,
            # top_k=40,
            # top_p=0.80,
            # grammar="json"
        )["choices"][0]["message"]["content"]
        json_objects = extract_json_objects(q_a)
        for object in json_objects:
            if "Answer" in object and "Question" in object:
                if(object["Question"] == context3):
                    print("question is the same as the context :/")
                    break
                if float(sentence_similarity(object["Question"],context3)) >90:
                    print("question is too similar to the context :/")
                    break
                question3 = object["Question"]
                answer3 = object["Answer"]
                return question3,answer3
        counter_bailout_QA+=1
    print("did not Generat a good pair Question Answer, Retry from context Generation")
    return None,None
def generate_A(question3):
    Counter_Bailout_a = 0
    Errors = []
    while Counter_Bailout_a < BAILOUT:
        ca = llm.create_chat_completion(
            messages=query_helper.create_A(question3),
            max_tokens=4096,
            temperature=0.5,
            # top_k=40,
            # top_p=0.80,
            # grammar="json"
            )["choices"][0]["message"]["content"]
        json_objects = extract_json_objects(ca)
        for object in json_objects:
            if "Answer" in object:
                consistencyAnswer = object["Answer"]
                return consistencyAnswer
            if isinstance(object ,str):
                match = re.search(r'```json\n(.*?)```', object, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    parsed = json.loads(json_str)
                    if "Answer" in parsed:
                        return parsed["Answer"]
        Errors.append(ca)
        Counter_Bailout_a +=1
    print("Bailout Creation of question (smth went wrong in the format of the generation)")
    print(Errors)
    return None
def Get_Consistency(Question,Answer1,Answer2):
    counter = 0
    while counter < BAILOUT:
        ca = llm.create_chat_completion(
            messages=query_helper.Get_consistency(Answer1,Answer2,Question),
            max_tokens=4096,
            temperature=0.2,
            # top_k=40,
            # top_p=0.80,
            # grammar="json"
            )["choices"][0]["message"]["content"]
        print(ca)
        json_objects = extract_json_objects(ca)
        for object in json_objects:
            if "consistency" in object:
                consistencyAnswer = object["consistency"]
                score = object["score"]
                return consistencyAnswer,score
        counter +=1
    return None,None

llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=11000, n_threads=20, verbose=False)

query_helper = query()

counter = 0

df = pd.read_csv("Data/Filtered_Output.csv")
df = df[df["Tag"] == TAG]

New_datas = []

while counter < NUMBER_TO_GEN:
    print(f"_________________________________________ {counter}/{NUMBER_TO_GEN} ____________________________________________")
    random_rows = df.sample(n=2)
    context3 = None
    question3 = None
    answer3 = None
    consistencyAnswer = None
    Counter_Failure = 0
    try:
        context3 = generate_context(random_rows)
        if context3 is not None:
            while(Counter_Failure < BAILOUT):
                question3,answer3 = generate_QA(random_rows,context3)
                if question3 is not None:
                    consistencyAnswer = generate_A(question3)
                    if consistencyAnswer is not None:
                        validation,score = Get_Consistency(question3,answer3,consistencyAnswer)
                        print(validation,score)
                        if validation is not None:
                            if validation.lower() == "yes":
                                New_datas.append({"context":context3,"question":question3,"answer":answer3})
                                counter+=1
                                break
                            else:
                                Counter_Failure+=1
                                print("did not pass verification, redo the question answer on the same subject")
                                if(Counter_Failure >= BAILOUT):
                                    print("pair question answer did not work produce good answer, bailout")
                else:
                    print("Regenerating Context ...")
                    Counter_Failure = BAILOUT
    except Exception as e:
        print(e)
        print("Retry Whole process")
        
                            
with open("data.json", "w") as file:
    json.dump(New_datas, file, indent=4)
print("Saved to data.json!")