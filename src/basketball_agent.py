import os
import re
import requests
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
if not api_key:
    print("警告: 未找到 API_KEY 环境变量")
else:
    print("已检测到 API_KEY 环境变量")

ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

class DashScopeEmbeddingsCustom(Embeddings):
    def __init__(self, model="text-embedding-v2", api_key=None):
        self.model = model
        self.api_key = api_key
        self.url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

    def embed_documents(self, texts):
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": self.model, "input": texts}
        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Embedding API error: {response.status_code} {response.text}")
        result = response.json()
        return [item["embedding"] for item in result["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

RAG_TEMPLATE = """你是一个篮球知识专家，精通篮球规则和球员信息。

参考资料：
{context}

对话历史：
{history}

问题：{question}

请用专业严谨的语言回答。如果问题与篮球无关，请回答："抱歉，我是篮球专家，只能回答篮球相关问题。"

回答要求：
- 必须根据参考资料回答，不要依赖外部知识
- 参考资料中没有的信息，直接说"参考资料中未找到相关信息"，不要猜测
- 涉及球员时，必须注明球员所属联赛（CBA/NBA/其他）
- 如果对话历史中已经讨论过相关话题，可以结合历史记录连贯地回答
- 追问时注意指代明确
- 重要信息用**粗体**标注"""

PLAYER_KW = ["球员", "球星", "谁", "哪个队", "效力", "身高", "体重", "位置", "号码", "几号",
             "后卫", "前锋", "中锋", "控球", "得分后卫", "小前锋", "大前锋", "CBA", "NBA",
             "詹姆斯", "库里", "杜兰特", "乔丹", "科比", "姚明", "易建联", "郭艾伦", "周琦"]


class BasketballRuleAgent:
    def __init__(self):
        self.vectorstore = None
        self.chain = None
        self.llm = None

    def _build_chain(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "..", "data")
        chroma_dir = os.path.join(current_dir, "..", "chroma_db")

        loader = DirectoryLoader(
            path=data_dir, glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        print(f"加载了 {len(documents)} 个文档")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        print(f"分割为 {len(splits)} 个片段")

        embeddings = DashScopeEmbeddingsCustom(model="text-embedding-v2", api_key=api_key)

        self.vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings,
            persist_directory=chroma_dir
        )
        print(f"向量存储就绪: {self.vectorstore._collection.count()} 个向量")

        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        print("创建 LLM...")
        self.llm = ChatOpenAI(
            model="qwen-plus", temperature=0.3,
            base_url=ALIYUN_BASE_URL, api_key=api_key
        )

        prompt = PromptTemplate(input_variables=["context", "question", "history"], template=RAG_TEMPLATE)

        # context 需要从输入 dict 中提取 question，再传给 retriever
        self.chain = (
            {"context": RunnableLambda(lambda x: x["question"]) | retriever, "question": lambda x: x["question"], "history": lambda x: x.get("history", "")}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("Chain 构建完成")

    def _ensure_init(self):
        if self.chain is None:
            self._build_chain()

    def _get_k(self, question):
        return 8 if any(kw in question for kw in PLAYER_KW) else 5

    def _filter_sources(self, docs, question):
        q_chars = set(re.sub(r'[？?，。！!、\s]', '', question))
        scored = []
        for doc in docs:
            content = doc.page_content
            overlap = sum(1 for c in q_chars if c in content)
            if overlap > 0:
                scored.append((overlap, content[:200]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored[:3]]

    def _build_history_str(self, history):
        if not history:
            return ""
        lines = []
        for msg in history[-6:]:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}：{msg['content']}")
        return "\n".join(lines)

    def get_answer(self, question, history=None):
        if history is None:
            history = []

        if not api_key:
            return {"result": f"模拟回答: 您的问题是 '{question}'。", "source_documents": []}

        try:
            self._ensure_init()
            if not self.chain:
                return {"result": "智能体初始化失败，请检查 API_KEY。", "source_documents": []}

            history_str = self._build_history_str(history)
            answer = self.chain.invoke({"question": question, "history": history_str})

            k = self._get_k(question)
            docs = self.vectorstore.similarity_search(question, k=k)
            source_docs = self._filter_sources(docs, question)

            return {"result": answer, "source_documents": source_docs}
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            return {"result": f"发生错误: {str(e)}", "source_documents": []}


def main():
    print("篮球规则智能问答助手")
    print("=====================")
    agent = BasketballRuleAgent()
    while True:
        question = input("你: ")
        if question.lower() == 'exit':
            print("再见！")
            break
        result = agent.get_answer(question)
        print(f"助手: {result['result']}\n")


if __name__ == "__main__":
    main()
