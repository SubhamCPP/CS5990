# -------------------------------------------------------------------------
# AUTHOR: Subham Panda
# FILENAME: similarity
# SPECIFICATION: Performed term frequency and finding cosine similarity
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 30 Minutes
# -----------------------------------------------------------*/

# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"

# Use the following words as terms to create your document-term matrix
# [soccer, favorite, sport, like, one, support, olympic, games]
# --> Add your Python code here
# Terms to be used in the document-term matrix
terms = ["soccer", "favorite", "sport", "like", "one", "support", "olympic", "games"]
def doc_to_vec(doc, terms):
    return [doc.split().count(term) for term in terms]
# Creating document vectors
doc_vectors = [doc_to_vec(doc, terms) for doc in [doc1, doc2, doc3, doc4]]

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors only
# Use cosine_similarity([X, Y, Z]) to calculate the pairwise similarities between multiple vectors
# --> Add your Python code here
# Calculating pairwise cosine similarities
similarity_matrix = cosine_similarity(doc_vectors)

# Ignoring diagonal and upper triangular matrix values
np.fill_diagonal(similarity_matrix, 0)

# Finding the pair of documents with the highest cosine similarity
max_sim = np.max(similarity_matrix)
max_sim_indices = np.where(similarity_matrix == max_sim)

# Print the highest cosine similarity following the information below
# The most similar documents are: doc1 and doc2 with cosine similarity = x
# --> Add your Python code here
# Printing the most similar documents and their cosine similarity
doc_indices = list(zip(max_sim_indices[0], max_sim_indices[1]))[0]  # Taking the first pair if multiple pairs have the same max similarity
print(f"The most similar documents are: doc{doc_indices[0]+1} and doc{doc_indices[1]+1} with cosine similarity = {max_sim:.3f}")
