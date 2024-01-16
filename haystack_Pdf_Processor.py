def main():
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
    from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor

    doc_dir="./Textbooks/CrackingTheCodingInterview.pdf"
    # all_docs = convert_files_to_docs(dir_path=doc_dir)
    converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
    doc_pdf = converter.convert(file_path=doc_dir, meta=None)[0]

    from haystack.nodes import PreProcessor
    preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,    #prevents sentences from being cut off
    )
    docs = preprocessor.process([doc_pdf])
    print(f"n_docs_output: {len(docs)}")

    from haystack import Document

    batch_size = 256
    total_doc_count = len(docs)

    counter = 0
    embedded_Docs = []
    for doc in docs:

        embedded_Docs.append(doc)
        counter += 1
        if counter % batch_size == 0 or counter == total_doc_count:
            embeds = retriever.embed_documents(embedded_Docs)
            for i, doc in enumerate(embedded_Docs):
                doc.embedding = embeds[i]
            document_store.write_documents(embedded_Docs)
            embedded_Docs.clear()
        if counter == total_doc_count:
            break




if __name__=="__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()