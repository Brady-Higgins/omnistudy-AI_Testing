#uses the MyUPDf library to split a textbook into 500 word chunks
#More advanced versions may split via paragraph and not exact chunks

#Issues:
#currently splits words
#experiment with text sizes, lots of overlap right now, leading to very ineffficient uploads

import fitz

class Process_PDF:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text_from_pdf(self):
        doc = fitz.open(self.pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        return text

    def preprocess_text(self, text):
        # Implement text preprocessing steps (e.g., removing unwanted characters, handling line breaks)
        # You might use regular expressions or string manipulation functions for cleaning.
        cleaned_text = text  # Placeholder, replace with actual preprocessing logic
        return cleaned_text

    def segment_text(self, text, max_chunk_length, stride):
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), stride)]
        return chunks

# # Example usage:
# pdf_processor = Process_PDF("path/to/your/pdf/file.pdf")
# text = pdf_processor.extract_text_from_pdf()
# cleaned_text = pdf_processor.preprocess_text(text)
# text_chunks = pdf_processor.segment_text(cleaned_text)
