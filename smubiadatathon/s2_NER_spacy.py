import spacy
import pandas as pd
from spacy import displacy

# Load the spaCy model
NER = spacy.load("en_core_web_sm")

# Define the NER function to get the Doc object
def spacy_large_ner(document):
    return NER(document)  # Return the Doc object

# Read the CSV file (adjust the file name and column name accordingly)
df = pd.read_csv("processed_data.csv")

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    document_text = row["clean_text"] 
    doc = spacy_large_ner(document_text)  # Process the text with spaCy

    # Visualize the entities
    print(f"Visualizing entities for row {index + 1}...")
    displacy.serve(doc, style="ent")
