"""
LLM 驱动的分片相关性打分器（通用、无硬编码）
"""

from typing import List, Dict, Any

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.llm import LLMClient

logger = get_logger(__name__)


class ChunkScorer:
    """使用 LLM 对分片与子问题的相关性进行打分（0.0~1.0）。"""

    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm = llm_client

    def score(self, sub_question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对每个分片进行打分，返回包含 `llm_score` 字段的分片副本列表。
        """
        if not chunks:
            return []

        # 构造紧凑上下文，避免提示过长
        snippet_lines = []
        for idx, ch in enumerate(chunks[:6], 1):  # 最多取 6 个候选
            text = (ch.get("content") or ch.get("text") or "").strip().replace("\n", " ")
            text = text[:700]
            snippet_lines.append(f"[{idx}] {text}")

        prompt = (
            "You are a neutral relevance judge. For the given sub-question, score each passage's relevance\n"
            "on a continuous scale between 0.0 (irrelevant) and 1.0 (highly relevant).\n"
            "Return only a JSON list of numbers in the same order as the passages. No extra text.\n\n"
            f"Sub-question: {sub_question}\n\n"
            f"Passages:\n" + "\n".join(snippet_lines)
        )

        try:
            raw = self.llm.generate(prompt, temperature=0.0)
            # 清理响应，移除<think>标签等内容
            import re
            cleaned_raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            
            # 尝试提取JSON数组
            import json
            json_match = re.search(r'\[[\d\s.,]+\]', cleaned_raw)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                scores = json.loads(cleaned_raw)
                
            if not isinstance(scores, list) or len(scores) == 0:
                raise ValueError("LLM did not return a valid list")
            
            logger.info(f"LLM 相关性评分成功: {scores}")
        except Exception as e:
            # 解析失败时，使用启发式回退分数
            logger.warning(f"LLM 相关性评分解析失败: {e}，使用启发式回退分数")
            scores = []
            for i, ch in enumerate(chunks[:6]):
                # 父分片通常包含更多上下文，给予稍高分数
                if str(ch.get("chunk_id", "")).startswith("parent_"):
                    base_score = 0.7
                else:
                    base_score = 0.5
                
                # 根据相似度调整分数
                similarity = ch.get("similarity", 0.0)
                adjusted_score = min(1.0, base_score + similarity * 0.3)
                scores.append(adjusted_score)

        # 合并分数到分片副本
        scored = []
        for i, ch in enumerate(chunks[:len(scores)]):
            ch_copy = dict(ch)
            try:
                ch_copy["llm_score"] = float(scores[i])
            except Exception:
                # 父分片默认给更高分数
                if str(ch.get("chunk_id", "")).startswith("parent_"):
                    ch_copy["llm_score"] = 0.7
                else:
                    ch_copy["llm_score"] = 0.5
            scored.append(ch_copy)

        # 其余未参与评分的候选，给低分
        for ch in chunks[len(scored):]:
            ch_copy = dict(ch)
            if str(ch.get("chunk_id", "")).startswith("parent_"):
                ch_copy["llm_score"] = 0.6
            else:
                ch_copy["llm_score"] = 0.3
            scored.append(ch_copy)

        # 按 llm_score 降序，相同分数时父分片优先
        scored.sort(key=lambda x: (x.get("llm_score", 0.0), 1 if str(x.get("chunk_id", "")).startswith("parent_") else 0), reverse=True)
        return scored 