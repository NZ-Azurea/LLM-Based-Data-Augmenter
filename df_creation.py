import pandas as pd
from bs4 import BeautifulSoup

def custom_html_parser(text):
    if isinstance(text, str):
        soup = BeautifulSoup(text, "html.parser")
        
        # Remplacement des balises <code> par ``r et </code> par ``
        for code_tag in soup.find_all("code"):
            code_tag.insert_before("\n\n```r\n")
            code_tag.insert_after("\n```\n\n")
            code_tag.unwrap()
        
        return soup.get_text()
    return text  # Retourner tel quel si ce n'est pas du texte

# 🔹 Load CSV files with selected columns
questions_df = pd.read_csv("Data/Questions.csv", usecols=["Id", "Title", "Body", "Score"])
answers_df = pd.read_csv("Data/Answers.csv", usecols=["ParentId", "Score", "IsAcceptedAnswer", "Body"])
tags_df = pd.read_csv("Data/Tags.csv", usecols=["Id", "Tag"])

# 🔹 Rename columns for clarity
questions_df.rename(columns={"Score": "QuestionScore", "Body": "QuestionBody"}, inplace=True)
answers_df.rename(columns={"Score": "AnswerScore", "Body": "AnswerBody"}, inplace=True)

# 🔹 Merge DataFrames
merged_df = questions_df.merge(answers_df, left_on="Id", right_on="ParentId", how="outer")
merged_df = merged_df.merge(tags_df, left_on="Id", right_on="Id", how="outer")

# 🔹 Drop redundant column
merged_df.drop(columns=["ParentId"], inplace=True)
merged_df.drop(columns=["Id"], inplace=True)

# 🔹 Remove rows with any missing (NaN) values
merged_df.dropna(inplace=True)

# 🔹 Filter rows where AnswerScore > 0
filtered_df = merged_df[merged_df["AnswerScore"] > 0]
filtered_df = merged_df[merged_df["QuestionScore"] > 0]
# 🔹 Compute medians
question_median = filtered_df["QuestionScore"].median()
answer_median = filtered_df["AnswerScore"].median()

# Appliquer la fonction sur les colonnes contenant du HTML
filtered_df["QuestionBody"] = filtered_df["QuestionBody"].apply(custom_html_parser)
filtered_df["AnswerBody"] = filtered_df["AnswerBody"].apply(custom_html_parser)

# 🔹 Save the cleaned and filtered data (optional)
filtered_df.to_csv("Data/Filtered_Output.csv", index=False)

# 🔹 Display results
print(f"Median Question Score: {question_median}")
print(f"Median Answer Score: {answer_median}")

print(filtered_df.head())
print(filtered_df.shape[0])

# nombre_lignes = (filtered_df['Tag'] == 'vector').sum()
# print(nombre_lignes)