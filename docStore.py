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

from dotenv import load_dotenv
import os

class docStore:
    def __init__(self):
        self.pinecone_api_key=""
        self.huggingface_api_token=""
    def initialize_index(self,index_name):
        pass
    def Assign_API_Keys(self):
        #Load environment variables from .env file
        
        #overide = true just forces a reload on the .env file in case api key changes
        load_dotenv(override=True)
        # Access the API key
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.huggingface_api_token = os.getenv("HUGGING_FACE_API_TOKEN")
    














import pinecone
from dotenv import load_dotenv
import os

# Load environment variables from .env file
#overide = true just forces a reload on the .env file in case api key changes
load_dotenv(override=True)

# Access the API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
huggingface_api_token = os.getenv("HUGGING_FACE_API_TOKEN")

#Initialize the pinecone index     
pinecone.init(      
	api_key=pinecone_api_key,      
	environment='gcp-starter'      
)      
index = pinecone.Index('haystack')

#Initialize pinecone doc store
from haystack.document_stores import PineconeDocumentStore
# document_store = PineconeDocumentStore(
#      api_key=api_key,
#      index='haystack-lfqa',
#      similarity="cosine",
#      embedding_dim=768
# )

document_store = PineconeDocumentStore(
    api_key=api_key
    pinecone_index='haystack'
    index='haystack'
    similarity="cosine",
    embedding_dim=768
)

document_store.
#Textbook Extraction to make digestable for docu store
from TextBookExctraction import Process_PDF
#max_chunk_length is the max token length of each vector within the database
#stride refers to the step taken to find the middle of each vector. 
#If stride is 2 and if max_length is 3, we move 2 steps forwards and each vector will contain 3 tokens with an overlap of 1
# [1,2,3] , [3,4,5], [5,6,7], ... , [n-1,n,n+1]            with each array referring to a chunk/vector
pdf_processor = Process_PDF(pdf_path="./Textbooks/CrackingTheCodingInterview.pdf")
text = pdf_processor.extract_text_from_pdf()
cleaned_text = pdf_processor.preprocess_text(text)
text_chunks = pdf_processor.segment_text(cleaned_text, max_chunk_length=500, stride=200)
print(len(text_chunks))
 
#Initialize retriever model
from haystack.nodes import EmbeddingRetriever
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    model_format="sentence_transformers"
)


#
batch_size = 256
total_doc_count = len(text_chunks)

counter = 0
docs = []
for d in text_chunks:
    docs.append(d)
    counter += 1
    if counter % batch_size == 0 or counter == total_doc_count:
        # Index documents to Pinecone
        document_store.write_documents(docs)
        docs.clear()

from haystack.document_stores import pinecone

