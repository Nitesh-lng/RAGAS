from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

class PDFDocumentLoader:

    def __init__(self,file_path):
        self.file_path=file_path

    def load(self):
        if not self.file_path.exists():
            raise FileNotFoundError('File Not Found!')
        if self.file_path.suffix.lower() != '.pdf':
            raise ValueError('File extension different')

        loader=PyPDFLoader(str(self.file_path))
        document= loader.load()

        if not document:
            raise ValueError('File is empty')

        return document

