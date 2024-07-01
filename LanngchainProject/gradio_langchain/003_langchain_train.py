from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_KEY')

# ------------------------------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------------------------------

# PDF 파일 로드
loader = PyPDFLoader('./docs/wallstreet.pdf')
document = loader.load()
print('총 길이 : ', len(document))
print(document[0].page_content[:200])     # 내용 추출

# ------------------------------------------------------------------------------------
# 데이터 분할
# ------------------------------------------------------------------------------------

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)
texts = text_splitter.split_documents(document)
print('문자열 길이 : ', len(texts))
print('<내용>')
print(texts[2:4])

# ------------------------------------------------------------------------------------
# 임베딩 정보 저장
# ------------------------------------------------------------------------------------
persist_directory = 'db'

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory)

# ------------------------------------------------------------------------------------
# 결과 출력하기
# ------------------------------------------------------------------------------------
print(vectordb.persist())