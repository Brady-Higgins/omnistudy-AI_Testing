#experimental implementation of pinecone document store
#Pinecone Document Store is a cloud based vector database which can be used in conjunction with retrieval models
#to provide context to nlps.
#It works by converting documents (in our case maybe 100 word fragments of a textbook) into a database of vectors
#These vectors show a relation (think of how google finds content most simliar to your search) 
#A retrieval model then grabs the vectors most simliar to a user's query and supplies them
#This powerful tool will be used in conjunction with an nlp model to provide a context specific answer to a question.


#I did this within a virtual env and stored my api key as an env variable. Neccessary installs are below
#pip install pinecone-client
#pip install farm-haystack

import os
import pinecone

# Retrieve the API key from the environment variable
api_key = os.environ.get('PINECONE_API_KEY')
if api_key is None:
    raise ValueError("Pinecone API key not found in environment variables.")


#Initialize pinecone doc store
pinecone.init(
    api_key=api_key,
    environment='gcp-starter'
)
index = pinecone.Index('testing')


#Textbook Extraction to make digestable for docu store
from TextBookExctraction import Process_PDF
#max_chunk_length is the max token length of each vector within the database
#stride refers to the step taken to find the middle of each vector. 
#If stride is 2, we move 2 steps forwards and if max_length is 3, each vector will contain 3 tokens with an overlap of 1
# [1,2,3] , [3,4,5], [5,6,7], ... , [n-1,n,n+1]            with each array referring to a chunk/vector
pdf_processor = Process_PDF(pdf_path="./Textbooks/CrackingTheCodingInterview.pdf")
text = pdf_processor.extract_text_from_pdf()
cleaned_text = pdf_processor.preprocess_text(text)
text_chunks = pdf_processor.segment_text(cleaned_text, max_chunk_length=384, stride=128)
print(len(text_chunks))
 