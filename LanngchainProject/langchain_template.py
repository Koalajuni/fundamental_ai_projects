import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')


# 질문을 템플릿으로 구성
template = '{country}의 수도는 뭐야 ?'

# 채팅 객체생성
llm = ChatOpenAI(
    temperature=1.2,
    max_tokens=2048,
    model_name='gpt-3.5-turbo',
)


# template settings
prompt = PromptTemplate(template=template, input_variables=['country'])

llm_chain = LLMChain(prompt=prompt, llm=llm)

# 실제 처리하는 과정:
# rst = llm_chain.run(country='미국')

# print(rst)

inputs = [
    {'country': '미국'},
    {'country': '대한민국'},
    {'country': '일본'},
    {'country': '스페인'},
]

# 한꺼번에 결과를 ㅐㅂ열로 처리
rst = llm_chain.apply(inputs)
print(rst)
