from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone as PineconeStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('7067cceb-9d01-4123-8e3c-cf060249ad85')


# print(PINECONE_API_KEY)

extracted_data = load_pdf()
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()



index_name="medical-chatbot"

#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_existing_index(index_name, embeddings)
