"""
DMQR-RAG + LightRAG + AT-RAG 迭代系统
流程：
1. 使用 DMQR-RAG 对原始查询进行多策略改写
2. 对改写后的查询提取低级关键词（实体/具体词）和高级关键词（主题/概念）
3. 使用 LightRAG 进行实体关系及文档检索
4. 使用 at-rag中的CoT思想生成答案，并对答案进行评估：
   - 若答案合格（无幻觉 + 有效解答问题），直接返回
   - 若不合格，at-rag 重写查询 → 回到步骤1迭代
5. 达到最大迭代次数后强制返回最后一次答案
"""
from dotenv import load_dotenv
load_dotenv()
import asyncio
import os
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from config import logger
from rag.at_rag import cot_self_rag
from lightrag import LightRAG
from lightrag.llm import qwen_max_complete
from lightrag.operate import _get_node_data, _get_edge_data, combine_contexts
from lightrag.prompt import PROMPTS
from rag.prompts import DMQR_PROMPT
from rag.constants import DMQR_STRATEGY_TYPES

WORKING_DIR = "./dickens"
light_rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=qwen_max_complete,  # Use gpt_4o_mini_complete LLM model
    llm_model_kwargs={"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}
)


from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APITimeoutError
from lightrag import LightRAG, QueryParam



@dataclass
class Config:
    # LLM
    model: str = "qwen-max"
    api_key: str = ""
    base_url: str = ""
    # 嵌入模型
    embed_model: str = "text-embedding-v3"
    embed_dim: int = 2048
    # LightRAG
    working_dir: str = "./lightrag_workspace"
    lightrag_query_mode: str = "hybrid"
    # 迭代控制
    max_iter: int = 5


_async_client: Optional[AsyncOpenAI] = None


def get_async_client(config: Config) -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(
            api_key=config.api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=config.base_url or os.environ.get("OPENAI_API_BASE", ""),
        )
    return _async_client


async def llm_call(
    prompt: str,
    config: Config,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.0,
) -> str:
    client = get_async_client(config)
    try:
        resp = await client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except (APIConnectionError, RateLimitError, APITimeoutError) as e:
        logger.error(f"LLM API 错误: {e}")
        raise
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        raise


def safe_json_extract(text: str, key: str, default = {}) -> Any:
    """从 LLM 返回文本中安全提取 JSON 字段"""
    # 先尝试找 markdown 代码块
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = code_block.group(1) if code_block else text
    # 找第一个 JSON 对象
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data.get(key, default)
        except json.JSONDecodeError:
            pass
    return default


async def dmqr_rewrite(query: str, config: Config) -> List[str]:
    """
    DMQR-RAG 多策略改写
    返回改写后查询列表（包含原始查询）
    """
    rewrites: List[str] = [query]  # 始终保留原始查询

    async def _call(query: str) -> Tuple[str, str]:
        prompt = DMQR_PROMPT.format(query=query)
        result = await llm_call(prompt, config)
        return result.strip()

    results = await asyncio.gather(*[_call(query)], return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"DMQR 改写出错: {result}")
            continue
        texts = result.split("\n")

        for res in texts:
            if not res or res == query:
                continue
            res = res.lstrip(" ").rstrip(" ")
            strag_str_list = res.split(":")
            if len(strag_str_list) < 2:
                continue
            if strag_str_list[0] not in DMQR_STRATEGY_TYPES:
                continue
            rewrites.append(strag_str_list[1])

    # 去重
    seen = set()
    unique = []
    for q in rewrites:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


async def extract_keywords(
    queries: List[str], config: Config
) -> Tuple[List[str], List[str]]:
    """
    同时提取低级关键词和高级关键词
    Returns: (all_low_kws, all_high_kws)
    """

    raw_keywords = await asyncio.gather(
        *[llm_call(PROMPTS["keywords_extraction"].format(query=query, examples=PROMPTS["keywords_extraction_examples"], language="Chinese"), config) for query in queries]
    )
    #低级关键词和高级关键词合并
    all_low_kws = []
    all_high_kws = []
    low_kws: List[List[str]] = [[]+safe_json_extract(raw_keyword, "low_level_keywords", []) for raw_keyword in raw_keywords]
    for low_kw in low_kws:
        all_low_kws.extend(low_kw)
    high_kws: List[List[str]] = [safe_json_extract(raw_keyword, "high_level_keywords", []) for raw_keyword in raw_keywords]
    for high_kw in high_kws:
        all_high_kws.extend(high_kw)
    logger.info(f"  [低级关键词] {all_low_kws}")
    logger.info(f"  [高级关键词] {all_high_kws}")
    return list(set(all_low_kws)), list(set(all_high_kws))


@dataclass
class State:
    """流水线中间状态"""
    iteration: int = 0
    current_query: str = ""
    rewritten_queries: List[str] = field(default_factory=list)
    low_kws: List[str] = field(default_factory=list)
    high_kws: List[str] = field(default_factory=list)


@dataclass
class FinalResult:
    """流水线最终输出"""
    answer: str
    total_iterations: int
    final_query: str
    rewritten_queries: List[str]
    low_kws: List[str]
    high_kws: List[str]


class IntegratedRAGPipeline:
    """
    DMQR-RAG + LightRAG + AT-RAG
    完整流程:
    step 1: DMQR 多策略改写
    step 2: 低级/高级关键词提取
    step 3: LightRAG检索
    step 4: AT-RAG中生成思考过程，将思考过程与问题和上下文拼接用于生成答案，然后评估 通过 → 返回答案 不通过 → AT-RAG 重写查询 → 回到 Step 1 迭代
    """

    def __init__(self, config: Config, rag: Optional[LightRAG] = None):
        self.config = config
        self.rag: Optional[LightRAG] = rag

    async def run(self, question: str) -> FinalResult:
        """
        执行完整的迭代 RAG
        Args:
            question: 原始用户问题
        Returns:
            FinalResult 包含最终答案和调试信息
        """

        state = State(current_query=question)

        logger.info("=" * 60)
        logger.info(f"开始处理查询: {question}")
        logger.info("=" * 60)

        for iteration in range(1, self.config.max_iter + 1):
            state.iteration = iteration
            logger.info(f"\n{'─' * 50}")
            logger.info(f"迭代 #{iteration} | 当前查询: {state.current_query[:80]}")

            # step 1: DMQR 改写
            logger.info("step 1: DMQR 查询改写")
            rewrites = await dmqr_rewrite(state.current_query, self.config)
            state.rewritten_queries = rewrites
            logger.info(f"  共生成 {len(rewrites)} 个改写查询")
            logger.info(rewrites)

            # step 2: 关键词提取
            logger.info("step 2: 关键词提取")
            low_kws, high_kws = await extract_keywords(rewrites, self.config)
            state.low_kws = low_kws
            state.high_kws = high_kws
            ll_keywords_str = ", ".join(low_kws) if low_kws else ""
            hl_keywords_str = ", ".join(high_kws) if high_kws else ""

            # step 3: LightRAG 检索
            logger.info("step 3: LightRAG 检索与回答")

            (
                ll_entities_context,
                ll_relations_context,
                ll_text_units_context,
            ) = await _get_node_data(
                ll_keywords_str,
                self.rag.chunk_entity_relation_graph,
                self.rag.entities_vdb,
                self.rag.text_chunks,
                QueryParam(mode="hybrid")
            )

            (
                hl_entities_context,
                hl_relations_context,
                hl_text_units_context,
            ) = await _get_edge_data(
                hl_keywords_str,
                self.rag.chunk_entity_relation_graph,
                self.rag.relationships_vdb,
                self.rag.text_chunks,
                QueryParam(mode="hybrid")
            )

            entities_context, relations_context, text_units_context = combine_contexts(
                [hl_entities_context, ll_entities_context],
                [hl_relations_context, ll_relations_context],
                [hl_text_units_context, ll_text_units_context],
            )
            docs = f"""
            -----Entities-----
            ```csv
            {entities_context}
            ```
            -----Relationships-----
            ```csv
            {relations_context}
            ```
            -----Sources-----
            ```csv
            {text_units_context}
            ```
            """
            gene_ans_or_new_question = cot_self_rag.run_pipeline(question=question, documents=docs)
            if "better_question" in gene_ans_or_new_question:
                state.current_query = gene_ans_or_new_question["better_question"]
                continue

            best_eval = FinalResult(
                answer=gene_ans_or_new_question["generation"],
                total_iterations=iteration,
                final_query=state.current_query,
                rewritten_queries=state.rewritten_queries,
                low_kws=state.low_kws,
                high_kws=state.high_kws,
            )

            return best_eval

        return FinalResult(
            answer="无法生成满意的答案，请尝试调整问题。",
            total_iterations=state.iteration,
            final_query=state.current_query,
            rewritten_queries=state.rewritten_queries,
            low_kws=state.low_kws,
            high_kws=state.high_kws,
        )


async def run_pipeline(
    question: str,
    config: Optional[Config] = None,
) -> FinalResult:
    """

    Args:
        question: 用户问题
        config: 配置
    Returns:
        FinalResult
    """
    if config is None:
        config = Config(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_API_BASE", ""),
        )

    pipeline = IntegratedRAGPipeline(config=config, rag=light_rag)

    result = await pipeline.run(question)
    return result


if __name__ == "__main__":
    asyncio.run(run_pipeline(question="牛顿第二定律的原始表述是什么"))
