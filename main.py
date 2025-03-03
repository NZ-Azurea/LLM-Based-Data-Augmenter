from llama_cpp import Llama
import pandas as pd
from query_class_helper import query
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

MODEL_PATH = "E:\Code_Perso\Python\model\Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
TAG = "vector"
NUMBER_TO_GEN = 4500
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

# context1="Accessing the Last Element of a Nested Vector in R"
# context2="Explain the quantile() function in R"
# question1 = """Suppose I have a vector that is nested in a dataframe one or two levels.  Is there a quick and dirty way to access the last value, without using the <code>length()</code> function?  Something ala PERL's <code>$#</code> special var?
# So I would like something like:
# <code>dat$vec1$vec2[$#]\n</code>
# instead of
# <code>dat$vec1$vec2[length(dat$vec1$vec2)]\n</code>"""
# answer1 ="""use variables in the outer function instead of global variables. This gets you the best of both approaches: you're not mutating global state, and you're not copying a big wad of data. If you have to exit early, just return the partial results.
# (See the ""Scope"" section in the R manual: <a href=""http://cran.r-project.org/doc/manuals/R-intro.html#Scope"" rel=""noreferrer"">http://cran.r-project.org/doc/manuals/R-intro.html#Scope</a>)"""
# question2="""I've been mystified by the R quantile function all day.
# I have an intuitive notion of how quantiles work, and an M.S. in stats, but boy oh boy, the documentation for it is confusing to me.
# From the docs:
# <blockquote>
#   Q[i](p) = (1 - gamma) x[j] + gamma
#   x[j+1],
# </blockquote>
# I'm with it so far.  For a type <em>i</em> quantile, it's an interpolation between x[j] and x [j+1], based on some mysterious constant <em>gamma</em>
# <blockquote>
#   where 1 &lt;= i &lt;= 9, (j-m)/n &lt;= p &lt;
#   (j-m+1)/ n, x[j] is the jth order
#   statistic, n is the sample size, and m
#   is a constant determined by the sample
#   quantile type. Here gamma depends on
#   the fractional part of g = np+m-j.
# </blockquote>
# So, how calculate j?   m?
# <blockquote>
#   For the continuous sample quantile
#   types (4 through 9), the sample
#   quantiles can be obtained by linear
#   interpolation between the kth order
#   statistic and p(k)
#   p(k) = (k - alpha) / (n - alpha - beta
#   + 1),
#   where α and β are constants determined
#   by the type. Further, m = alpha + p(1
#   - alpha - beta), and gamma = g.
# </blockquote>
# Now I'm really lost.  p, which was a constant before, is now apparently a function.
# So for Type 7 quantiles, the default...
# <blockquote>
#   Type 7
#   p(k) = (k - 1) / (n - 1). In this case, p(k) = mode[F(x[k])]. This is used by S.
# </blockquote>
# Anyone want to help me out?  In particular I'm confused by the notation of p being a function and a constant, what the heck <em>m</em> is, and now to calculate j for some particular <em>p</em>.
# I hope that based on the answers here, we can submit some revised documentation that better explains what is going on here.
# <a href="https://svn.r-project.org/R/trunk/src/library/stats/R/quantile.R" rel="noreferrer">quantile.R source code</a>
# or type:  quantile.default"""
# answer2="""You're understandably confused.  That documentation is terrible.  I had to go back to the paper its based on (Hyndman, R.J.; Fan, Y. (November 1996). ""Sample Quantiles in Statistical Packages"". <em>American Statistician</em> 50 (4): 361–365. <a href=""http://dx.doi.org/10.2307%2F2684934"" rel=""nofollow noreferrer"">doi:10.2307/2684934</a>) to get an understanding.  Let's start with the first problem.
# <blockquote>
#   where 1 &lt;= i &lt;= 9, (j-m)/n &lt;= p &lt;  (j-m+1)/ n, x[j] is the jth order statistic, n is the sample size, and m is a constant determined by the sample quantile type. Here gamma depends on the fractional part of g = np+m-j.
# </blockquote>
# The first part comes straight from the paper, but what the documentation writers omitted was that <code>j = int(pn+m)</code>.  This means <code>Q[i](p)</code> only depends on the two order statistics closest to being <code>p</code> fraction of the way through the (sorted) observations.  (For those, like me, who are unfamiliar with the term, the ""order statistics"" of a series of observations is the sorted series.)
# Also, that last sentence is just wrong.  It should read
# <blockquote>
#   Here gamma depends on the fractional part of np+m, g = np+m-j
# </blockquote>
# As for <code>m</code> that's straightforward.  <code>m</code> depends on which of the 9 algorithms was chosen.  So just like <code>Q[i]</code> is the quantile function, <code>m</code> should be considered <code>m[i]</code>.  For algorithms 1 and 2, <code>m</code> is 0, for 3, <code>m</code> is -1/2, and for the others, that's in the next part.
# <blockquote>
#   For the continuous sample quantile types (4 through 9), the sample quantiles can be obtained by linear interpolation between the kth order statistic and p(k):
#   p(k) = (k - alpha) / (n - alpha - beta + 1), where α and β are constants determined by the type. Further, m = alpha + p(1 - alpha - beta), and gamma = g.
# </blockquote>
# This is really confusing.  What the documentation calls <code>p(k)</code> is not the same as the <code>p</code> from before.  <code>p(k)</code> is the <a href=""http://mathworld.wolfram.com/PlottingPosition.html"" rel=""nofollow noreferrer"">plotting position</a>.  In the paper, the authors write it as <code>p</code><sub><code>k</code></sub>, which helps.  Especially since in the expression for <code>m</code>, the <code>p</code> is the original <code>p</code>, and the <code>m = alpha + p * (1 - alpha - beta)</code>.  Conceptually, for algorithms 4-9, the points (<code>p</code><sub><code>k</code></sub>, <code>x[k]</code>) are interpolated to get the solution (<code>p</code>, <code>Q[i](p)</code>).  Each algorithm only differs in the algorithm for the <code>p</code><sub><code>k</code></sub>.
# As for the last bit, R is just stating what S uses.
# The original paper gives a list of 6 ""desirable properties for a sample quantile"" function, and states a preference for #8 which satisfies all by 1.  #5 satisfies all of them, but they don't like it on other grounds (it's more phenomenological than derived from principles).  #2 is what non-stat geeks like myself would consider the quantiles and is what's described in wikipedia.
# BTW, in response to <a href=""https://stackoverflow.com/questions/95007/explain-the-quantile-function-in-r/397303#397303"">dreeves answer</a>, Mathematica does things significantly differently.  I think I understand the mapping.  While Mathematica's is easier to understand, (a) it's easier to shoot yourself in the foot with nonsensical parameters, and (b) it can't do R's algorithm #2.  (Here's <a href=""http://mathworld.wolfram.com/Quantile.html"" rel=""nofollow noreferrer"">Mathworld's Quantile page</a>, which states Mathematica can't do #2, but gives a simpler generalization of all the other algorithms in terms of four parameters.)"""

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
    
