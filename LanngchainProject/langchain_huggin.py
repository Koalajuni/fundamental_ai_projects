import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGING_KEY')

model_id = 'mistalai/Mistal-7B-v0.1'

# 질의응답
q = "Who is Son Heung Min?"

# 템플릿
template = """ Question: {question}

Answer: 
"""

prompt = PromptTemplate(template=template, input_variables=['question'])

# llm 모델을 허깅페이스 모델로 교체
llm = HuggingFaceHub(
    repo_id=model_id,
    model_kwargs={
        "temperature": 0.2,
        "max_length": 128,
    }
)

# 모델 갹체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

rst = llm_chain.run(question=q)
print(rst)
