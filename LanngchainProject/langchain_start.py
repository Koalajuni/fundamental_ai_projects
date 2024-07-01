from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from openai import OpenAI
from dotenv import load_dotenv


client = OpenAI()


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')


print(client.models.list())

# LangChain 채팅 객체생성
llm = ChatOpenAI(
    temperature=0,  # 창의성(헛소리)
    max_tokens=2048,  # 최대 토큰 수
    mode_name='gpt3.5-turbo',
)


# 질의내용
qus = '대한민국 수도에 대해 알려줘'

# 결과
print(f'[결과]:{llm.predict(qus)}')
