"""
LLM 驱动的通用查询扩展器（PRF）

- 不使用问题/领域硬编码
- 先用原始问题做一次初检索（由调用方提供 top 文档）
- 使用 LLM 基于问题与上下文生成通用的改写/扩展检索子句
- 可用于信息完整性补全（建议额外检索方向）
"""

from __future__ import annotations

from typing import List, Dict, Any

from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.llm import LLMClient

logger = get_logger(__name__)


class LLMPRFExpander:
	"""基于 LLM 的伪相关反馈查询扩展器（通用、无硬编码）。"""

	def __init__(self, config: Config, llm_client: LLMClient):
		self.config = config
		self.llm = llm_client

	def expand(self, question: str, top_docs: List[Dict[str, Any]], max_expansions: int = 3) -> List[str]:
		"""基于问题与检索到的 top 文档，生成若干通用的检索扩展子句。

		返回若干短语或查询句，每行一个，不包含任何领域硬编码。
		"""
		context_snippets = []
		for d in (top_docs or [])[:5]:
			text = (d.get("content") or d.get("text") or "").strip()
			if text:
				context_snippets.append(text[:500])

		prompt = (
			"You are a retrieval assistant. Given a user question and several retrieved passages,\n"
			"produce up to N alternative search queries that could retrieve more complementary\n"
			"and precise evidence.\n\n"
			"Constraints:\n"
			"- Be domain-agnostic. Do not assume domain-specific terms beyond what appears in the passages.\n"
			"- Prefer neutral paraphrases, synonyms, unit variations, and generic formulations.\n"
			"- Include potential missing aspects implied by the question but not yet evidenced in passages.\n"
			"- Return one query per line; no numbering, no extra text.\n\n"
			f"N={max_expansions}\n\n"
			f"Question: {question}\n\n"
			f"Passages:\n- " + "\n- ".join(context_snippets)
		)

		raw = self.llm.generate(prompt, temperature=0.1)
		lines = [l.strip() for l in raw.splitlines() if l.strip()]
		# 截断到 max_expansions，去重
		seen = set()
		expanded = []
		for q in lines:
			if q not in seen:
				seen.add(q)
				expanded.append(q)
				if len(expanded) >= max_expansions:
					break
		return expanded

	def assess_completeness(self, question: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""利用 LLM 对信息完整性进行通用评估，输出可能缺失的方面与建议。
		不包含任何领域硬编码。
		"""
		context_snippets = []
		for d in (docs or [])[:6]:
			text = (d.get("content") or d.get("text") or "").strip()
			if text:
				context_snippets.append(text[:600])

		prompt = (
			"You are a retrieval QA completeness checker.\n"
			"Given a user question and the current supporting passages, identify which aspects\n"
			"of the question appear to be covered and which aspects are likely missing.\n\n"
			"Return a concise JSON with keys: covered_aspects, missing_aspects, suggestions.\n"
			"- covered_aspects: short phrases of what seems answered\n"
			"- missing_aspects: short phrases of what seems not evidenced yet\n"
			"- suggestions: generic search directions to fill the gaps (domain-agnostic)\n\n"
			f"Question: {question}\n\n"
			f"Passages:\n- " + "\n- ".join(context_snippets)
		)

		raw = self.llm.generate(prompt, temperature=0.1)
		# 简单的容错解析：若非严格 JSON，也尝试行分割
		result: Dict[str, Any] = {
			"covered_aspects": [],
			"missing_aspects": [],
			"suggestions": []
		}
		try:
			import json
			parsed = json.loads(raw)
			if isinstance(parsed, dict):
				result.update({k: parsed.get(k, result[k]) for k in result.keys()})
		except Exception:
			# 回退：逐行提取
			lines = [l.strip("- ") for l in raw.splitlines() if l.strip()]
			# 简化：前 3 行视为覆盖，后 3 行视为缺失，再后续为建议
			result["covered_aspects"] = lines[:3]
			result["missing_aspects"] = lines[3:6]
			result["suggestions"] = lines[6:9]
		return result 