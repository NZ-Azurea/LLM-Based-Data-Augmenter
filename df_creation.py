import pandas as pd

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

# 🔹 Save the cleaned and filtered data (optional)
filtered_df.to_csv("Data/Filtered_Output.csv", index=False)

# 🔹 Display results
print(f"Median Question Score: {question_median}")
print(f"Median Answer Score: {answer_median}")

print(filtered_df.head())
print(filtered_df.shape[0])