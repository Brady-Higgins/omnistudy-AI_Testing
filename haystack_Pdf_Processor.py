from haystack.utils import convert_files_to_docs
#env + system imports
from dotenv import load_dotenv
import os
#pincone
import pinecone  
from haystack.document_stores import PineconeDocumentStore
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, EmbeddingRetriever
from haystack import Pipeline


doc_dir="./Textbooks/CrackingTheCodingInterview.pdf"
all_docs = convert_files_to_docs(dir_path=doc_dir)

from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
doc_txt=converter(all_docs,meta=None)[0]

load_dotenv(override=True)
# Access the API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
huggingface_api_token = os.getenv("HUGGING_FACE_API_TOKEN")

#Initialize the pinecone index
index_name='haystack'
pinecone.init(      
api_key=pinecone_api_key,      
environment='gcp-starter'      
)      
index = pinecone.Index(index_name=index_name)

#Initialize the haystack document store object
document_store = PineconeDocumentStore(
api_key=pinecone_api_key,
pinecone_index=index,
similarity="cosine",
embedding_dim=768
)
from haystack.nodes import PreProcessor
preprocessor = PreProcessor(
clean_empty_lines=True,
clean_whitespace=True,
clean_header_footer=False,
split_by="word",
split_length=100,
split_respect_sentence_boundary=True,
)
docs_default = preprocessor.process([doc_txt])