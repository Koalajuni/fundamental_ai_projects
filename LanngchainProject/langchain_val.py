import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')


# 질문을 템플릿으로 구성
template = '{city1}과 {city2}의 거리는 얼마나 되지?'

# 채팅 객체생성
llm = ChatOpenAI(
    temperature=1.4,
    max_tokens=2048,
    model_name='gpt-3.5-turbo',
)

# template settings
prompt = PromptTemplate(template=template, input_variables=['city1', 'city2'])
llm_chain = LLMChain(prompt=prompt, llm=llm)


rst = llm_chain.run(city1='서울', city2='목포')
print(rst)


# callback
# 프로세스가 동작이 되는동안 처리되는 내용(상황)을 계속 보여주는 효과를 주기 위해서
# -> 비동기방식 구현
# -> 멀티채팅 구현
