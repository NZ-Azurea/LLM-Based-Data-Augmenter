from pydantic import BaseModel
from jinja2 import Template
import json

import sys
sys.stdout.reconfigure(encoding='utf-8')  # Force UTF-8 encoding

Template_str_Context = """
[
    {"role": "system", "content": "You are a bot that specializes in data augmentation and outputs JSON format. You will receive two contexts and generate a new context under the key 'context' that is related to the same subject (don't output the other context.). The context must be short and can be either a question or a description."},
    {"role": "user", "content": {{ ("context 1: " ~ context1 ~ "\\ncontext 2: " ~ context2) | tojson }} }
]
"""

Template_str_QA = """[
    {"role": "system", "content": "You are a bot that specializes in data augmentation and outputs JSON format. You will generate a new very exhaustive question-answer pair under the keys 'Question' and 'Answer' based on the given context. Take inspiration from the Stack Overflow style of conversation (that will be provided in the example), create a little story or explaine how you got to the probleme before asking the qestion and don't ouput anything else than the json itself (what's inside the key must be just string). Two examples will be given by the user. Use tags like in the examples in the question answer pair generated"},
    {"role": "user", "content": {{ ("Example 1:\\ncontext: " ~ context1 ~ "\\nquestion: " ~ question1 ~ "\\nanswer: " ~ answer1 ~ "\\nExample 2:\\ncontext: " ~ context2 ~ "\\nquestion: " ~ question2 ~ "\\nanswer: " ~ answer2 ~ "\\n\\nNow generate based on the next context:\\ncontext: " ~ context3) | tojson }} }
]"""

Template_str_A = """[
    {"role": "system", "content": "You are a bot that specializes in data augmentation and outputs JSON format. You will generate a new very exhaustive answer from a given question under the keys 'Answer' based on the given context. Take inspiration from the Stack Overflow style of conversation (that will be provided in the example) and don't ouput anything else than the json itself (what's inside the key must be just the answer in a string format). Two examples will be given by the user."},
    {"role": "user", "content": {{ ("Example 1:\\ncontext: " ~ context1 ~ "\\nquestion: " ~ question1 ~ "\\nanswer: " ~ answer1 ~ "\\nExample 2:\\ncontext: " ~ context2 ~ "\\nquestion: " ~ question2 ~ "\\nanswer: " ~ answer2 ~ "\\n\\nNow generate based on the next context:\\ncontext: " ~ context3 ~ " \\n question:" ~ question3) | tojson }} }
]"""

Template_str_consistency = """
[
    {"role": "system", "content": "You are a bot that specializes in consistency checking and outputs JSON format. You will receive two different answers with its associated question and will determine whether they are consistent with each other, responding with 'Yes' or 'No' under the key 'consistency' and a consistency score between 0 and 1 under the key 'score'."},
    {"role": "user", "content": {{ ("1st answer: " ~ answer1 ~ "\\n2nd answer: " ~ answer2) | tojson }} }
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
    
    def create_context(self,context1,context2) -> dict:
        context1.replace('"', '\\"')
        context2.replace('"', '\\"')
        data={"context1":context1,"context2":context2}
        template = Template(Template_str_Context)
        messages = json.loads(template.render(**data))
        self.context1 = context1,
        self.context2 = context2
        return messages
    
    def create_QA(self,question1,question2,answer1,answer2,context3) -> dict:
        question1.replace('"', '\\"')
        question2.replace('"', '\\"')
        answer1.replace('"', '\\"')
        answer2.replace('"', '\\"')
        context3.replace('"', '\\"')
        data = {"context1":self.context1,"context2":self.context2,"context3":context3,"question1":question1,"question2":question2,"answer1":answer1,"answer2":answer2}
        template = Template(Template_str_QA)
        messages = json.loads(template.render(**data))
        self.context3 = context3
        self.question1=question1
        self.question2=question2
        self.answer1=answer1
        self.answer2=answer2
        return messages
        
    def create_A(self,question3) -> dict:
        question3.replace('"', '\\"')
        data = {"context1":self.context1,"context2":self.context2,"context3":self.context3,"question1":self.question1,"question2":self.question2,"answer1":self.answer1,"answer2":self.answer2,"question3":question3}
        template = Template(Template_str_A)
        messages = json.loads(template.render(**data))
        return messages
    
    def Get_consistency(self,Answer1,Answer2,Question) -> dict:
        Answer1.replace('"', '\\"')
        Answer2.replace('"', '\\"')
        Question.replace('"', '\\"')
        data = {"answer1":Answer1,"answer2":Answer2,"question":Question}
        template = Template(Template_str_consistency)
        messages = json.loads(template.render(**data))
        return messages
    
        
        
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

# queryhelper = query()

# print(queryhelper.create_context(context1=context1,context2=context2))
# print(queryhelper.create_QA(question1,question2,answer1,answer2,context2))
# print(queryhelper.create_A(question1))






