#experimental implementation of pinecone document store
#Pinecone Document Store is a cloud based vector database which can be used in conjunction with retrieval models
#to provide context to nlps.
#It works by converting documents (in our case maybe 100 word fragments of a textbook) into a database of vectors
#These vectors show a relation (think of how google finds content most simliar to your search) 
#A retrieval model then grabs the vectors most simliar to a user's query and supplies them
#This powerful tool will be used in conjunction with an nlp model to provide a context specific answer to a question.


#I did this within a virtual env and stored my api key as an env variable. Neccessary installs are below
#pip install pinecone-client
#pip install farm-haystack[pinecone]
#tqdm version problems, so had to edit tqdm/auto.py to ignore tqdm_asyncio

#env + system imports
from dotenv import load_dotenv
import os
#pincone
import pinecone  
from haystack.document_stores import PineconeDocumentStore
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, EmbeddingRetriever
from haystack import Pipeline

# #PDF Processing
from TextBookExctraction import Process_PDF

class context_QA:
    def __init__(self):
        self.pinecone_api_key=""
        self.huggingface_api_token=""
        self.document_store=None
        self.retriever=None
        self.search_pipe=None
        self.QA_pipeline=None
    def get_document_store(self,index_name):
        #Initialize the pinecone index
        pinecone.init(      
        api_key=self.pinecone_api_key,      
        environment='gcp-starter'      
        )      
        index = pinecone.Index(index_name=index_name)

        #Initialize the haystack document store object
        self.document_store = PineconeDocumentStore(
        api_key=self.pinecone_api_key,
        pinecone_index=index,
        similarity="cosine",
        embedding_dim=768
)
    def Assign_API_Keys(self):
        #Load environment variables from .env file
        # (overide = true) just forces a reload on the .env file in case api key changes
        load_dotenv(override=True)
        # Access the API key
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.huggingface_api_token = os.getenv("HUGGING_FACE_API_TOKEN")
    def extract_textbook(self,pdf_path):
        #Textbook Extraction to make digestable for docu store
        
        #max_chunk_length is the max token length of each vector within the database
        #stride refers to the step taken to find the middle of each vector. 
        #If stride is 2 and if max_length is 3, we move 2 steps forwards and each vector will contain 3 tokens with an overlap of 1
        # [1,2,3] , [3,4,5], [5,6,7], ... , [n-1,n,n+1]            with each array referring to a chunk/vector
        pdf_processor = Process_PDF(pdf_path)
        text = pdf_processor.extract_text_from_pdf()
        cleaned_text = pdf_processor.preprocess_text(text)
        text_chunks = pdf_processor.segment_text(cleaned_text, max_chunk_length=500, stride=400)
        return text_chunks
    def init_retriever(self,top_k):
        self.retriever = EmbeddingRetriever(
        document_store=self.document_store,
        embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
        model_format="sentence_transformers",
        top_k=top_k
        )
    def create_pipeline(self):
        #embedding retriever pipeline
        self.search_pipe = DocumentSearchPipeline(retriever=self.retriever)

        self.QA_pipeline = Pipeline()
        prompt ="""Synthesize a comprehensive answer from the following top_k most relevant paragraphs and the given question. 
                             Provide a clear and concise response that summarizes the key points and information presented in the paragraphs. 
                             Your answer should be in your own words and be no longer than 50 words. 
                             \n\n Paragraphs: {join(documents)} \n\n Question: {query} \n\n Answer:"""
        template = PromptTemplate(prompt=prompt,output_parser=AnswerParser())
        node = PromptNode(model_name_or_path="mistralai/Mistral-7B-v0.1",default_prompt_template=template,api_key=self.huggingface_api_token)
        self.QA_pipeline.add_node(component=node,name="prompt_node",inputs=["Query"])
    def train_retriever(self,pdf_path):
        #only ran if textbook is not already uploaded to pinecone
        from haystack import Document
        
        text_chunks = self.extract_textbook(pdf_path)
        batch_size = 256
        total_doc_count = len(text_chunks)

        counter = 0
        docs = []
        for d in text_chunks:
            doc = Document(
                content = d
            )
            docs.append(doc)
            counter += 1
            if counter % batch_size == 0 or counter == total_doc_count:
                embeds = self.retriever.embed_documents(docs)
                for i, doc in enumerate(docs):
                    doc.embedding = embeds[i]
                self.document_store.write_documents(docs)
                docs.clear()
            if counter == total_doc_count:
                break
    def QA_output(self,query):
        print(type(self.search_pipe))
        print(type(self.QA_pipeline))
        res=self.QA_pipeline.run(query=query,documents=self.search_pipe.run(query=query,params={"Retriever": {"top_k": 2}})['documents'])
        return res['answers']
    def initQA(self,index_name,top_k):
        self.Assign_API_Keys()
        self.get_document_store(index_name)
        self.init_retriever(top_k)
        self.create_pipeline()
        


#
#index_name = 'haystack'
#pdf_path="./Textbooks/CrackingTheCodingInterview.pdf"
#top_k=3







