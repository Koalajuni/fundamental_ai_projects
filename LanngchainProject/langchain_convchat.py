import os
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')


chat = ChatOpenAI()


rst = chat(
    [HumanMessage(content="다음을 영어로 번역해줘: 나는 파이썬을 좋아한다.")]
)

# print(rst)

# messages = [
#     SystemMessage(
#         content="너는 영어선생님인데, 고3 영어를 가르치는 선생님이야. 내 영어 질문에 한글로 해석을 해주고, 고3이 이해할 수 있는 수준으로 자세히 설명해줘"),
#     # 문의
#     HumanMessage(content="I am a boy")
# ]

# rst = chat(messages)

# print(rst)

while True:
    msg = input("끝말을 입력하세요: ")
    chatmsg = [
        SystemMessage(
            content="이제 나랑 끝말잇기 할거야, 처음 시작할 떄 단어를 이어줘"),
        # 문의
        HumanMessage(content=msg)
    ]

    rst = chat(chatmsg)
    print(rst)
