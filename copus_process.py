import os
import shutil

from lightrag import LightRAG, QueryParam
from lightrag.llm import qwen_max_complete
from dotenv import load_dotenv
load_dotenv()
#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
    shutil.move("book.txt", os.path.join(WORKING_DIR, "book.txt"))

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=qwen_max_complete,
    llm_model_kwargs={"base_url": os.environ.get("OPENAI_API_BASE", "")}
)

#语料处理
with open("./dickens/book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("牛顿第二定律的原始表述是什么？", param=QueryParam(mode="hybrid"))
)