# %%
# -------------------------------------------------------------------------
# AUTHOR: Subham Panda
# FILENAME: similarity
# SPECIFICATION: Performed term frequency and finding cosine similarity
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 30 Minutes
# -----------------------------------------------------------*/

# %%
# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# %%

# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"


# %%

# Use the following words as terms to create your document-term matrix
# [soccer, favorite, sport, like, one, support, olympic, games]
# --> Add your Python code here

# Defining the terms for the document-term matrix
terms = ["soccer", "favorite", "sport", "like", "one", "support", "olympic", "games"]

# Creating the CountVectorizer object with the predefined vocabulary
vectorizer = CountVectorizer(vocabulary=terms)

# Compiling all documents into a list for vectorization
documents = [doc1, doc2, doc3, doc4]

# Vectorizing the documents to create the document-term matrix
X = vectorizer.fit_transform(documents).toarray()
X

# %%

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
# --> Add your Python code here

# Calculating pairwise cosine similarities
cosine_similarities = cosine_similarity(X)
cosine_similarities


# %%


# Print the highest cosine similarity following the information below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
# --> Add your Python code here

# Initializing variables to store the indexes of the most similar documents and the highest similarity score
max_similarity = 0
doc_index1 = 0
doc_index2 = 0


# Iterating through the cosine similarity matrix to find the highest similarity score (excluding self-comparisons)
for i in range(len(cosine_similarities)):
    for j in range(i + 1, len(cosine_similarities)):
        if cosine_similarities[i][j] > max_similarity:
            max_similarity = cosine_similarities[i][j]
            doc_index1, doc_index2 = i, j

# Printing the most similar documents and their cosine similarity
print(f"The most similar documents are: doc{doc_index1 + 1} and doc{doc_index2 + 1} with cosine similarity = {max_similarity:.2f}")


# %%



