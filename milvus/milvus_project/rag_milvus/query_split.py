from llama_index.core.prompts import PromptTemplate
from llm import VolcengineLLM
import json
import time


def extract_json_from_markdown(text: str) -> str:
    # 去掉前后 ```json 和 ```
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-len("```")].strip()
    return text

def parse_query_to_json(query: str, llm) -> dict:
    # 定义few-shot示例
    few_shot_examples = [
        {
            "query": "宁德时代二季度毛利率为什么能逆势提升？这种趋势能维持吗？",
            "output": {
                "date": "2025年二季度",
                "date_range": "2025-04-01至2025-06-30",
                "institution": "无",
                "target_entity": "宁德时代",
                "industry": "动力电池",
                "authors": "无",
                "fund_codes": "无",
                "fund_names": "无",
                "topic": "宁德时代2025Q2毛利率逆势提升的原因分析及持续性评估",
                "keywords": "原材料成本,技术溢价,竞争格局",
                "report_type": "公司点评"
            }
        },
        {
            "query": "想了解中金彭文生和摩根士丹利邢自强最近三个月对港股互联网板块的看法，特别是美联储可能降息的影响",
            "output": {
                "date": "最近三个月",
                "date_range": "2025-04-09至2025-07-09",
                "institution": "中金公司/摩根士丹利",
                "target_entity": "港股互,网板块",
                "industry": "科技",
                "authors": "彭文生,邢自强",
                "fund_codes": "无",
                "fund_names": "无",
                "topic": "美联储降息预期对港股科技股估值修复路径的影响",
                "keywords": "无风险利率,DCF模型,流动性改善",
                "report_type": "宏观研究"
            }
        },
        {
            "query": "美国那个生物安全法案对药明康德去年业绩实际造成了多大冲击？",
            "output": {
                "date": "2024年度",
                "date_range": "无",
                "institution": "无",
                "target_entity": "药明康德,药明生物",
                "industry": "CXO",
                "authors": "无",
                "fund_codes": "无",
                "fund_names": "无",
                "topic": "美国生物安全法案对国内CXO企业2024年业绩影响的量化分析",
                "keywords": "地缘政治风险,订单可见度,产能利用率",
                "report_type": "专题研究"
            }
        },
        {
            "query": "高盛和大摩最近对人民币突然走强有什么新解读？下半年怎么看？",
            "output": {
                "date": "本月",
                "date_range": "2025-07-01至2025-07-09",
                "institution": "高盛,摩根大通",
                "target_entity": "人民币汇率",
                "industry": "外汇",
                "authors": "无",
                "fund_codes": "无",
                "fund_names": "无",
                "topic": "主要投行对近期人民币快速升值原因及下半年走势的最新观点",
                "keywords": "中美利差,跨境资本流动,央行干预",
                "report_type": "宏观研究"
            }
        }
    ]

    # 构建few-shot提示模板
    few_shot_prompt = "\n\n".join([
        f"查询: {example['query']}\n输出: ```json\n{json.dumps(example['output'], ensure_ascii=False, indent=2)}\n```"
        for example in few_shot_examples
    ])

    # 定义完整的提示模板
    prompt_template = PromptTemplate(
        """你是财经领域智能助手，任务是将用户查询解析为JSON格式的键值对，包含以下字段：
    - authors: 作者（若无明确作者，填“无”）
    - date: 具体日期（如“2023”或“2023-01”）
    - date_range: 日期范围（如“2022-2023”）
    - fund_codes: 基金代码
    - fund_names: 基金名称
    - industry: 行业（如“宏观”）
    - institution: 机构名称（如“国金证券”）
    - keywords: 关键词
    - report_type: 提取报告类型，如“行业研究”“策略研究”“宏观研究”“公司分析”“公司点评报告”“电子行业周报”“晨会纪要”等
    - target_entity: 目标实体（如“川式化债”）
    - topic: 主题描述
    
    请参考以下示例：
    {few_shot_examples}
    
    用户查询: {query}
    输出: ```json
    {{
      "authors": "无",
      "date": "无",
      "date_range": "无",
      "fund_codes": "无",
      "fund_names": "无",
      "industry": "无",
      "institution": "无",
      "keywords": "无",
      "report_type": "无",
      "target_entity": "无",
      "topic": "无"
    }}
```""".format(few_shot_examples=few_shot_prompt, query="{query}")
    )

    # 记录开始时间
    start_time = time.time()

    # 调用LLM解析查询
    parsing_prompt = prompt_template.format(query=query)
    response = llm.complete(parsing_prompt, system_prompt="你是财经领域智能助手，解析查询并输出JSON。")

    # 提取和解析JSON
    raw_text = extract_json_from_markdown(response.text.strip())
    try:
        parsed_json = json.loads(raw_text)
    except Exception as e:
        raise ValueError(f"JSON 解析失败: {e}, 原始输出为: {raw_text}")

    # 计算耗时
    elapsed = time.time() - start_time

    return {
        "original_query": query,
        "parsed_result": parsed_json,
        "elapsed_time": elapsed,
    }

if __name__ == "__main__":
    llm = VolcengineLLM(api_key='ff6acab6-c747-49d7-b01c-2bea59557b8d')
    query = "根据国金证券和华泰证券的宏观研究内容，分析美国通胀情况。"
    result = parse_query_to_json(query, llm)
    print(json.dumps(result, ensure_ascii=False, indent=2))