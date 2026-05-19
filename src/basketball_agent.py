import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 加载环境变量（优先使用系统环境变量）
load_dotenv()

# 确保能读取到 API_KEY
api_key = os.getenv("API_KEY")
if not api_key:
    print("警告: 未找到 API_KEY 环境变量")
    print("请在系统环境变量中设置 API_KEY")
    print("将使用模拟模式运行智能体...")
else:
    print("已检测到 API_KEY 环境变量")
    print("将使用在线模式运行智能体...")

# 阿里云百炼API配置
ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class BasketballRuleAgent:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.rag_prompt = None
        self.llm = None
        # 延迟初始化，直到首次调用get_answer时
        print("智能体初始化完成！")
    
    def initialize_agent(self):
        """初始化智能体，加载知识库并创建检索链"""
        if self.qa_chain:
            return
        
        print("正在初始化智能体...")
        try:
            # 获取当前文件所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 计算数据目录的绝对路径
            data_dir = os.path.join(current_dir, "..", "data")
            print(f"数据目录: {data_dir}")
            
            # 加载文档，指定UTF-8编码
            print("正在加载文档...")
            loader = DirectoryLoader(
                path=data_dir,
                glob="*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()
            print(f"加载了 {len(documents)} 个文档")
            
            # 分割文档
            print("正在分割文档...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            print(f"文档分割为 {len(splits)} 个片段")
            
            print("正在创建 Embeddings...")
            embeddings = DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=api_key
            )
            print("Embeddings 创建成功")

            print("正在创建向量存储...")
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=os.path.join(current_dir, "..", "chroma_db")
            )
            print(f"向量存储创建成功，共 {self.vectorstore._collection.count()} 个向量")

            print("正在创建检索器...")
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            print("检索器创建成功")

            print("正在创建LLM...")
            llm = ChatOpenAI(
                model="qwen-plus",
                temperature=0.3,
                base_url=ALIYUN_BASE_URL,
                api_key=api_key
            )
            print("LLM创建成功")

            print("正在构建 RAG 链...")
            rag_template = """你是一个篮球知识专家，精通篮球规则和球员信息。

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
- 追问时注意指代明确（"它"、"这个规则"等需要明确是什么）
- 重要信息用**粗体**标注"""
            self.rag_prompt = PromptTemplate(
                input_variables=["context", "question", "history"],
                template=rag_template
            )
            self.llm = ChatOpenAI(
                model="qwen-plus",
                temperature=0.3,
                base_url=ALIYUN_BASE_URL,
                api_key=api_key
            )
            self.qa_chain = (
                {"context": retriever, "question": RunnablePassthrough(), "history": lambda x: ""}
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )
            print("RAG 链构建成功")
            print("智能体初始化完成！")
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.qa_chain = None
    
    def _get_search_k(self, question):
        """根据问题类型动态调整检索数量，球员类问题多检索"""
        player_kw = ["球员", "球星", "谁", "哪个队", "效力", "身高", "体重", "位置", "号码", "几号",
                     "后卫", "前锋", "中锋", "控球", "得分后卫", "小前锋", "大前锋", "CBA", "NBA"]
        return 5 if any(kw in question for kw in player_kw) else 3

    def get_answer(self, question, history=None):
        """获取问题的回答"""
        if history is None:
            history = []

        if not api_key:
            return {"result": f"模拟回答: 您的问题是 '{question}'。这是一个模拟回答，因为没有设置 API_KEY。", "source_documents": []}

        try:
            self.initialize_agent()
            if not self.qa_chain:
                return {"result": "抱歉，智能体初始化失败，请检查 API_KEY 和网络连接后重试。", "source_documents": []}

            history_str = ""
            if history:
                history_lines = []
                for msg in history[-6:]:
                    role = "用户" if msg["role"] == "user" else "助手"
                    history_lines.append(f"{role}：{msg['content']}")
                history_str = "\n".join(history_lines)

            search_k = self._get_search_k(question)
            print(f"正在检索并生成回答... (k={search_k})")
            if history_str:
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": search_k}
                )
                history_chain = (
                    {"context": retriever, "question": RunnablePassthrough(), "history": lambda x: history_str}
                    | self.rag_prompt
                    | self.llm
                    | StrOutputParser()
                )
                answer = history_chain.invoke(question)
            else:
                answer = self.qa_chain.invoke(question)

            source_docs = []
            if self.vectorstore:
                docs = self.vectorstore.similarity_search(question, k=search_k)
                source_docs = [doc.page_content[:200] for doc in docs]

            return {"result": answer, "source_documents": source_docs}
        except Exception as e:
            print(f"获取回答失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"result": f"抱歉，获取回答时发生错误: {str(e)}", "source_documents": []}

def main():
    """主函数，提供命令行交互界面"""
    print("篮球规则智能问答助手")
    print("=====================")
    print("你可以询问任何关于篮球规则的问题，输入 'exit' 退出")
    print()
    
    agent = BasketballRuleAgent()
    
    while True:
        question = input("你: ")
        if question.lower() == 'exit':
            print("再见！")
            break
        
        try:
            result = agent.get_answer(question)
            answer = result["result"]
            print(f"助手: {answer}")
            print()
        except Exception as e:
            print(f"抱歉，发生错误: {str(e)}")
            print()

if __name__ == "__main__":
    main()