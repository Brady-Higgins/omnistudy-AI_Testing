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
