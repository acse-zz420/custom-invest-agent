import re
from typing import Any, List, Optional
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# （示例）模拟读入 Markdown 文本
# from text_txt.raw_md1 import markdown_text

table_pattern = re.compile(r"<html>\s*<body>\s*<table>", re.IGNORECASE)

def isolate_tables(chunks: List[str]) -> List[str]:
    """
    在每个 chunk 中查找 <html><body><table>...</table></body></html>。
    若找到，则拆分成：
      [ 文字前缀, 完整表格HTML, 文字后缀, ... ]
    多个表格则依序拆分。
    """
    table_re = re.compile(
        r"(<html>\s*<body>\s*<table>.*?</table>\s*</body>\s*</html>)",
        re.IGNORECASE | re.DOTALL
    )
    new_chunks = []

    for chunk in chunks:
        start_idx = 0
        for match in table_re.finditer(chunk):
            # 提取“表格前面的文字”
            if match.start() > start_idx:
                text_part = chunk[start_idx:match.start()].strip()
                if text_part:
                    new_chunks.append(text_part)

            # 表格本体
            table_part = chunk[match.start():match.end()].strip()
            new_chunks.append(table_part)

            start_idx = match.end()

        # 表格后面的文字
        if start_idx < len(chunk):
            tail = chunk[start_idx:].strip()
            if tail:
                new_chunks.append(tail)

    return new_chunks


########################################################################
# 复用已有的 _split_text_with_regex_from_end 与 ChineseRecursiveTextSplitter
########################################################################
def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s.strip() != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s",
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)

        # 清理多余的空行
        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip() != ""
        ]


def remove_unparseable_image_syntax(original_text: str) -> str:
    """
    移除 Markdown 中形如 ![](images/xxx.jpg) 的图片引用（大小写不敏感）。
    """
    pattern = r'!\[\]\(images/[^)]+\.jpg\)'
    text_no_images = re.sub(pattern, '', original_text, flags=re.IGNORECASE)

    return text_no_images


def remove_control_chars(text: str) -> str:
    """
    保留 \n \r \t，其它 ASCII 控制字符（0x00-0x1F 与 0x7F）去掉
    """
    control_chars = (
        ''.join(chr(c) for c in range(0x00, 0x20) if c not in (0x09, 0x0A, 0x0D)) +
        chr(0x7F)
    )
    control_char_re = f"[{re.escape(control_chars)}]"
    return re.sub(control_char_re, '', text)


def filter_invalid_chunks(chunks: List[str], threshold_number=5, count_same_word=True) -> List[str]:
    """
    过滤无效 chunks 的综合条件
      1) 首行是无效标题 → True
      2) 无效关键词出现次数 > 5 → True
      过滤掉则说明该 chunk 不保留。
    """
    INVALID_TITLES = [
        "免责声明", "重要声明", "评级说明", "投资评级说明", "分析师声明", "分析师介绍",
        "分析员个股评级定义", "研究团队", "分析师承诺", "本產品並無抵押品", "投资评级说明",
        "研究所", "特别声明", "法律声明", "公司声明及风险提示", "法律声明及风险提示", "风险提示",
        "分析员", "分析师", "执证编号", "信息披露", "相关报告", "相关研报", "相关研究报告"
    ]

    INVALID_KEYWORDS = [
        "首席", "证监会审核", "中国证监会", "电话", "邮箱", "传真", "邮编", "执业编号", "执业证书", "执证编号", "执业证书编号",
        "SAC", "从业资格号", "投资咨询号", "通讯录", "@", ".com", "证书编号", "联系人", "市场有风险",
        "股票评级", "行业评级", "业务资格", "研究员",
        "资格号", "咨询号", "投资评级", "评级说明", "本内容", "本报告", "本公司", "版权", "保留一切权利", "投资建议"
        "保留所有权利", "组团队介绍", "销售团队", "地址", "营业部", "Tel", "Fax", "公司注册号", "经营许可证"
    ]

    def _compile_title_patterns():
        """
        预编译无效标题正则
        """
        patterns = {}
        for title in INVALID_TITLES:
            pat = rf"""
                ^\s*                             
                (?:[#]+\s*)?                      
                (?:                              
                    (?:
                        \d+(?:\.\d+)?             
                      | [一二三四五六七八九十]+     
                      | [（(][一二三四五六七八九十\d]+[)）]
                    )
                    [\.、]?\s*                   
                )?                               
                {re.escape(title)}                
                (\s|$)                           
            """
            patterns[title] = re.compile(pat, re.VERBOSE | re.IGNORECASE)
        return patterns

    _TITLE_PATTERNS = _compile_title_patterns()

    def is_invalid_chunk(chunk: str) -> bool:
        """
        判定规则：
          1) 首行是无效标题 → True
          2) 无效关键词出现次数 > 5 → True
        """
        chunk_to_list = chunk.lstrip().splitlines()
        if not chunk_to_list:
            return False
        else:
            first_line = chunk_to_list[0]
            if re.sub(r"[\n\r\t\s]+", "", first_line) == '#':
                return False
            else:
                # ---- 1. 标题行检测 ----
                for pat in _TITLE_PATTERNS.values():
                    if pat.match(first_line):
                        return True

        # ---- 2. 关键词计数 ----
        # 2.1 统计有多少个不同关键词至少出现一次
        kw_different_cnt = sum(1 for kw in INVALID_KEYWORDS if kw in chunk)
        # 2.2 检查是否有任意一个关键词出现次数 > 5
        if count_same_word:
            for kw in INVALID_KEYWORDS:
                if chunk.count(kw) > threshold_number:
                    return True
            return kw_different_cnt > threshold_number
        else:
            pass

    return [chunk for chunk in chunks if not is_invalid_chunk(chunk)]


########################################################################
# 第 1 步: 用 MarkdownHeaderTextSplitter 执行初次文本切块
########################################################################
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)


########################################################################
# 判断是否“目录”的简易逻辑
########################################################################
import re

def is_toc_chunk(text: str) -> bool:
    """
    判断一个 Markdown chunk 是否为“目录/图表目录/正文目录”块并过滤掉。
    """
    heading_pattern = re.compile(r'^(#+)\s*(.*)', re.MULTILINE)
    headings = heading_pattern.findall(text)

    # 1. 标题行含“目录”关键字 → 视为目录
    for (sharp, title) in headings:
        normalized = re.sub(r'\s+', '', title)
        if "目录" in normalized:
            return True

    # 2. 文本整体不含“目录”，但检测“目录型结构”
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    if len(lines) == 1:
        return True

    # 3. 疑似目录行的正则（原有规则）
    toc_line_pattern = re.compile(
        r'^(\s*[\d一二三四五六七八九十]+[、\.])|'
        r'(\s*图\s*\d+：?)|'
        r'(\s*表\s*\d+：?)'
    )

    # 4. 新增规则：结尾为数字（页码），前面可有点、空格等
    page_number_pattern = re.compile(r'[\d\.·、…]+\s*$')

    directory_like_count = 0
    page_number_like_count = 0
    long_line_count = 0
    LONG_LINE_THRESHOLD = 120  # 目录行有时很长，阈值可适当放宽

    for ln in lines:
        if len(ln) > LONG_LINE_THRESHOLD:
            long_line_count += 1
        if toc_line_pattern.search(ln):
            directory_like_count += 1
        if page_number_pattern.search(ln):
            page_number_like_count += 1

    directory_like_ratio = directory_like_count / len(lines)
    page_number_like_ratio = page_number_like_count / len(lines)

    # 满足原目录规则
    if directory_like_ratio >= 0.6 and long_line_count <= 2:
        return True

    # 满足“页码结尾”目录规则
    if page_number_like_ratio >= 0.6 and long_line_count <= 2:
        return True

    return False


########################################################################
# 递归合并小于 threshold 的短 chunks
########################################################################
def merge_small_chunks(chunks: List[str], threshold: int = 400) -> List[str]:
    """
    递归合并小块逻辑：
      1) 当前 chunk 不含表格，且长度 < threshold；
      2) 下一个 chunk 不含表格；
      3) 满足上述条件则合并。合并后长度仍小于 threshold，则继续合并下一个，依此类推。
    """
    merged_chunks = []
    i = 0
    while i < len(chunks):
        current_chunk = chunks[i]
        # 如果当前块含表格，则不合并，直接放进 merged_chunks
        if table_pattern.search(current_chunk):
            merged_chunks.append(current_chunk)
            i += 1
        else:
            # 若当前块不含表格，则看长度是否小于阈值，若是则继续尝试往下递归合并
            while len(current_chunk) < threshold and (i + 1) < len(chunks):
                # 检查下一块是否含表格，若含就停止合并
                next_chunk = chunks[i + 1]
                if table_pattern.search(next_chunk):
                    break
                # 否则合并
                i += 1
                current_chunk += "\n" + next_chunk

            merged_chunks.append(current_chunk)
            i += 1
    return merged_chunks



########################################################################
# 若 chunk 中含 <html><body><table> 则不再切分，否则若长度大于 max_len 再次使用
# ChineseRecursiveTextSplitter 细分
########################################################################
text_splitter = ChineseRecursiveTextSplitter(
    keep_separator=True,
    is_separator_regex=True,
    chunk_size=500,
    chunk_overlap=0
)


def further_split(chunks: List[str], max_len: int = 1000) -> List[str]:
    new_chunks = []
    for chunk in chunks:
        # 判断是否包含 <html><body><table>
        if table_pattern.search(chunk):
            # 不执行进一步切分
            new_chunks.append(chunk)
        else:
            # 如果长度大于 max_len，则执行拆分
            if len(chunk) > max_len:
                sub_splits = text_splitter.split_text(chunk)
                new_chunks.extend(sub_splits)
            else:
                new_chunks.append(chunk)
    return new_chunks


def merge_small_chunks_once(chunks: List[str], threshold: int = 300) -> List[str]:
    """
    从后向前扫描进行合并，允许跨越表格块：

      1. 遍历顺序从列表尾部开始，对于每个 chunk：
         - 如果 chunk 为纯文本且字数小于 threshold，
           则向前（索引减小的方向）寻找第一个纯文本块（跳过中间所有表格块）；
         - 找到后，将当前 chunk 合并到该文本块后，并删除当前 chunk；
         - 如果没有找到符合条件的前方纯文本块，则保持该 chunk 原样（即使它字数不足）。

      2. 在扫描结束后，如果结果列表中的第一个 chunk 是纯文本且字数依然小于 threshold，
         则向后（索引增大的方向）寻找一个纯文本块，将其合并到第一个 chunk 中，并删除该被合并的块。

    该逻辑满足：跨越表格块进行合并，并处理了第一个纯文本块无前向合并候选的情况。
    """
    # 复制一份列表，方便就地删除处理
    result = chunks[:]
    i = len(result) - 1
    while i > 0:
        current_chunk = result[i]
        # 如果当前 chunk 是纯文本且字数不足阈值
        if not table_pattern.search(current_chunk) and len(current_chunk) < threshold:
            # 从当前位置向前寻找第一个纯文本块（跳过所有表格块）
            j = i - 1
            while j >= 0:
                if not table_pattern.search(result[j]):
                    # 合并：将当前 chunk 追加到找到的纯文本块后面
                    result[j] = result[j] + "\n" + current_chunk
                    # 删除当前 chunk
                    del result[i]
                    # 跳转到 j 继续向前扫描
                    i = j
                    break
                j -= 1
            else:
                # 如果未能找到前面的纯文本块，保持当前块原样
                i -= 1
        else:
            i -= 1

    # 最后，对首个 chunk 进行额外处理：
    # 如果第一个 chunk 是纯文本且字数不足阈值，则向后寻找最近的纯文本块进行合并
    if result and not table_pattern.search(result[0]) and len(result[0]) < threshold:
        for k in range(1, len(result)):
            if not table_pattern.search(result[k]):
                result[0] = result[0] + "\n" + result[k]
                del result[k]
                break

    return result

########################################################################
# 自定义切块管线
########################################################################
def custom_chunk_pipeline(text: str) -> List[tuple]:

    # 0) 文本清洗
    text = remove_unparseable_image_syntax(text)
    text = remove_control_chars(text)

    # 1) 使用 MarkdownHeaderTextSplitter 进行初次切块
    md_nodes = markdown_splitter.split_text(text)

    if len(md_nodes) >= 2:
        # 2) 过滤目录
        chunks = [n.page_content for n in md_nodes if not is_toc_chunk(n.page_content)]

        # 3) 过滤无效块
        chunks = filter_invalid_chunks(chunks)

        # 3.5) 提取纯表格
        chunks = isolate_tables(chunks)

        # 4) 合并过短的块（递归合并至 >=400 或到底）
        chunks = merge_small_chunks(chunks, threshold=400)

        # 5) 对仍然过长的块再次递归切分
        chunks = further_split(chunks, max_len=1000)

        # 6) 执行完再次切分后，再合并一次过短的块，若最后一块 < 300 字则合并到上一个块
        chunks = merge_small_chunks_once(chunks, threshold=300)

        # 7) 再次过滤
        chunks = filter_invalid_chunks(chunks, threshold_number=3)


    else:
        # 若切出的 chunk 不足 2 个，则用备用流程
        chunks = text_splitter.split_text(text)
        # 过滤目录
        chunks = [c for c in chunks if not is_toc_chunk(c)]
        # 过滤无效块
        chunks = filter_invalid_chunks(chunks)

    # 最后：为每个 chunk 做类型标记
    # 若含表格（<html><body><table>）→ 标记为 1，否则标记为 0
    chunks_list = []
    for ck in chunks:
        if table_pattern.search(ck):
            chunks_list.append((1, ck))
        else:
            chunks_list.append((0, ck))

    return chunks_list


########################################################################
# 使用示例 (仅供参考)
########################################################################
if __name__ == "__main__":
    text = r"""# 广发证券（000776.SZ）  

        # 或为当前最具投资价值的券商标的  

        广发证券股权结构整体稳定、经营管理层经验丰富，业绩改善向好无虞，是当之无愧的头部优质券商。然而对比头部同业，公司 PB 估值长期偏低，与基本面不符。在交投情绪高涨的市场环境中、券商股权变更浪潮的行业背景下，维持公司“强烈推荐”投资评级。  

        ❑ 公司业绩预计改善。1）广发证券“大资管”业务领先，控股子公司广发基金、参股公司易方达基金在行业内头部“交椅”稳固，尤其在被动化投资趋势下易方达凭借声誉优势、规模优势和投研实力有望持续扩宽、加深公司在资管领域的竞争优势。2）财富管理业务持续闪亮，金融科技赋能直指投资者交易痛点、堵点，传统经纪基本盘稳固；公司战略性打造、培育专业投顾团队，加之广州投顾业务生态建设持续完善，强劲的内核力量和良好的外部环境有望加速财富管理转型升级。3）交易及机构业务稳健，公司聚焦做市业务、很好理顺了“功能性”和“盈利性”之间的关系，投研长期保持行业前列、屡获殊荣。此外，公司投行业务发展进入稳定期、收入加速爬坡中；国际化布局持续深入。  

        ❑ 稳定的股权结构和经营团队为公司稳健经营提供坚实保障。吉林敖东、辽宁成大和中山公用 25 年来一直位列前三大股东，经营管理团队在证券、金融和经济相关领域的经历平均约 28 年、在公司平均任职期限超过 18 年，股东结构的长期稳定和经验丰富的管理层是公司具备高度凝聚力和战斗力、不断穿越周期、突破发展瓶颈、奠定行业地位的重要支撑。近日，公司第二大股东辽宁成大授权管理层处置所持有的不超过广发证券总股本 $3\%$ 的 A 股股份，引发市场对公司股权变更的遐想，但授权事项并非确定性执行方案、相关事宜仍待进一步观望。  

        ❑ 维持“强烈推荐”投资评级。展望后市，资本市场仍处于监管呵护的大周期内，伴随新一轮科技投资浪潮兴起，中国资产价值迎来重估，券商并购重组“下半场”开启在即，市场风偏持续向好，流动性充裕、赚钱效应充足，券商业绩、估值均有望改善。对比头部同业，目前广发证券 PB 估值较低，与公司潜在的业绩改善空间不匹配，整体安全边际较高。我们预计 24/25/26 年广发证券归母净利润分别为 82/101/118 亿，同比 $+18\%/+24\%/+16\%$ 。给定公司目标价 19.3 元，对应 1.1 倍 PB 目标，空间约为 $24\%$ ，维持强烈推荐目标。  

        ❑ 风险提示：政策力度或有效性不及预期，市场萎靡或投资情绪低迷，业务费率下行拖累收入下滑，公司市占率提升不及预期，合规风险。  

        # 强烈推荐 （维持）  

        总量研究/非银行金融目标估值：19.30 元当前股价：15.52 元  

        # 基础数据  

        总股本（百万股） 7621已上市流通股（百万股） 5919总市值（十亿元） 118.3流通市值（十亿元） 91.9每股净资产（MRQ） 18.7ROE（TTM） 5.6资产负债率 $80.8\%$ 主要股东 吉林敖东药业集团股份有限主要股东持股比例 $16.44\%$ % 1m 6m 12m绝对表现 1 31 15相对表现 -2 13 -1资料来源：公司数据、招商证券  

        ![](images/3031f25057ec0ded4a93451d7c6360b4585b55b7530c731496596fd6b9e6b0c3.jpg)  

        财务数据与估值  
        股价表现   


        <html><body><table><tr><td>会计年度</td><td>2022</td><td>2023</td><td>2024E</td><td>2025E</td><td>2026E</td></tr><tr><td>营业总收入(百万元)</td><td>25132</td><td>23300</td><td>27016</td><td>31093</td><td>34248</td></tr><tr><td>同比增长</td><td>-26.6%</td><td>-7.3%</td><td>16.0%</td><td>15.1%</td><td>10.1%</td></tr><tr><td>营业利润(百万元)</td><td>10448</td><td>8795</td><td>10586</td><td>12932</td><td>15097</td></tr><tr><td>同比增长</td><td>-30.5%</td><td>-15.8%</td><td>20.4%</td><td>22.2%</td><td>16.7%</td></tr><tr><td>归母净利润(百万元)</td><td>7929</td><td>6978</td><td>8199</td><td>10130</td><td>11794</td></tr><tr><td>同比增长</td><td>-26.9%</td><td>-12.0%</td><td>17.5%</td><td>23.6%</td><td>16.4%</td></tr><tr><td>每股收益(元)</td><td>1.02</td><td>0.83</td><td>1.08</td><td>1.33</td><td>1.55</td></tr><tr><td>PE</td><td>15.2</td><td>18.7</td><td>14.4</td><td>11.7</td><td>10.0</td></tr><tr><td>PB</td><td>1.1</td><td>1.0</td><td>1.0</td><td>0.9</td><td>0.8</td></tr></table></body></html>

        资料来源：公司数据、招商证券  

        # 相关报告  

        1、《广发证券（000776）一资管变中求进，自营表现突出》2024-11-012、《广发证券（000776）一业绩整体承压，资管核心优势维持，投行修复亮眼》2023-04-02  

        郑积沙 S1090516020001zhengjisha@cmschina.com.cn许诗蕾 研究助理xushilei@cmschina.com.cn  

        # 正文目录  

        一、 公司业绩预计改善  
        1、 “大资管”品牌效应显著  
        2、 投顾转型持续推进，财管业务做大做强 53、 交易机构业务相对稳健，衍生品聚焦做市. 64、 投行业绩爬坡加速  
        5、 增资香港子公司，发力海外业务 . 8稳定的股权结构和经验丰富的管理层为公司穿越周期提供坚实保障... 9三、 投资建议与风险提示 . 101、 行业景气度及盈利预测. . 102、 公司经营业绩展望及盈利预测 113、 投资建议.. 114、 风险提示， . 12  

        # 图表目录  

        图 1：广发证券资管收入及收入贡献  
        图 2：指数基金持有 A 股市值超过主动权益基金持有 A 股市值. 5图 3：易方达基金非货币 ETF 资产净值及环比增速 5图4：广发证券财富管理收入及收入贡献.. . 5图 5：广发证券代理交易金额及市占率 6图 6：广发证券代销非货保有规模及市占率 . 6图7：广发证券交易及机构业务收入及收入贡献.. 7图 8：IPO 募集资金排名：广发证券 vs 头部同业  
        图 9：债券承销金额排名：广发证券 vs 头部同业  
        图 10：24Q3 投行手续费净收入及同比：广发证券 vs 头部同业 . 8图 11：广发香港（控股）营业总收入及净利润 . 8图 12：广发证券股权结构 ... 9图 13：广发证券历史 PE Band . 12图 14：广发证券历史 PB Band . 12  

        表 1：广发证券管理层主要人员及工作经历 9  
        表2：证券行业景气度预测. 11  
        表 3：广发证券分业务预测. 11  
        表 4：损益表 （单位：百万元） . 13  
        表 5：资产负债表（单位：百万元） 14  

        # 一、公司业绩预计改善  

        # 1、“大资管”品牌效应显著  

        资管业务闪亮，是公司业绩的重要支撑。广发证券投资管理业务划分为三大子板块：通过广发资管、广发期货以及广发资管（香港）开展的资产管理业务、通过控股子公司广发基金和参股公司易方达基金开展的公募基金业务、通过广发信德和广发投资（香港）及其下属机构开展的私募基金业务。2020 年来，乘大公募时代的东风、享大财富管理时代的红利，易方达和广发基金实现管理规模的迅速扩张、进入行业第一梯队；同期，公司投资管理收入贡献大幅提高，2022 年一度逼近 $40\%$ 。  

        ![](images/e6911d80a1a1c6372b1623fd3c734e510800735fa5ce2347ef473dc022e661e4.jpg)  
        图 1：广发证券资管收入及收入贡献  
        资料来源：公司公告、招商证券  

        指数化投资浪潮涌动，多元化优势筑牢易方达基金在 ETF 赛道“护城河”。新“国九条”、《促进资本市场指数化投资高质量发展行动方案》等政策红利持续释放。中证 A500 ETF、科创综指 ETF 等创新产品相继上市，ETF 管理费率持续走低，推动 ETF 深受投资者的青睐、成为资产配置的优选。截至 24 年底，指数基金持有 A 股市值达 3.2 万亿，高出主动权益基金 A 股持仓市值 $26\%$ 。而易方达长期积累的市场声誉和投资者信任更易赢得资金青睐，规模效应持续摊薄ETF 市场营销和流动性支持成本，投研优势打开差异化、创新性产品创设空间，加深、扩宽了易方达在 ETF 赛道的竞争优势。截至 24 年底，易方达非货币 ETF资产净值达 5893 亿，同比增速达 $126\%$ 。此外，受益于重要机构投资者战略性增持宽基 ETF、托举市场，易方达沪深 300ETF 规模迅速扩张、成为市场上第二只规模站上 2 千亿元台阶的 ETF。往后看，伴随 ETF 渗透稳步提升，易方达有望持续扩大业务优势、坐稳头部交椅，公司“大资管”招牌有望继续闪亮。  

        ![](images/09a51c89a93717367a319a1050b778d81c0863cbbc3292b92395cce0c50190a2.jpg)  
        图 2：指数基金持有 A 股市值超过主动权益基金持有 A股市值  
        资料来源：Wind、招商证券  

        ![](images/ba9fad055aea6ead0898eb435c61693b3e3e326b244f51a27f6ee2af95a9a065.jpg)  
        图 3：易方达基金非货币 ETF 资产净值及环比增速  

        资料来源：Wind、招商证券  

        # 2、投顾转型持续推进，财管业务做大做强  

        财富管理业务韧性凸显。广发证券财富管理业务涵盖财富管理及经纪业务、融资融券业务、回购交易业务以及融资租赁业务，其中财富经纪业务为“顶梁柱”。长期以来，广发证券根植粤港澳大湾区、以研究为渠道、以客户需求为基础、以数智化一体化为引擎，为各类客户提供涵盖交易服务、投资理财、财富配置等全方位、个性化解决方案。近年来，二级景气度偏低、一级总量控制的背景下，公司财富管理业务发挥了业绩“压舱石”的作用，24H1 财管收入 52.2 亿元，同比增长 $2.4\%$ ，收入贡献比例为 $44.3\%$ ，同比增长 5.8pct。  

        ![](images/f24224022b4a800c0df200186407d42f3b5dc1ff49c7b4a7177947b17bb44072.jpg)  
        图 4：广发证券财富管理收 $\mathrm{\land}$ 及收入贡献  
        资料来源：公司公告、招商证券  

        金科赋能，传统经纪市占率稳定。作为行业内率先推动互联网对客服务平台数字化、数智化转型的券商，广发证券始终践行“以客户为中心”的理念、致力于全方位提升客户交易体验。2023 年，为满足适应客户群体代际转换、满足年轻投资者对数字化交易工具和便捷投资方式的需求，公司发布全新 Z 世代 APP、强化易淘金平台用户体验；机构客户服务方面，公司全面整合资源、推出“广发智汇”机构综合服务平台，提升机构客户服务能力、完善机构客户服务体系。长期  

        # m  

        耕耘和平台效应累积下，互联网平台提供的服务基本覆盖公司个人客户，截至2023 年底，易淘金用户数达 4460 万、同比增长 $9.4\%$ ；传统经纪业务市占率稳定在 $4.6\%$ 左右。  

        强劲的内核力量叠加优质的产业生态环境，推动财管业务做大做强。广发证券坐拥一支“了解客户、精通产业、掌握配置、严守风险和运用科技”五大核心能力、超过 4400 人的投顾团队，旗下两家公募基金投研能力强劲、产品设计能力突出、能够构建全资产类别的综合产品服务方案，同时，公司积极参与广州投顾产业“ $1+4+N^{\gamma}$ 框架建设、优先享受广州投顾业态建设成果。多重优势加持下，公司代销业务领先多数同业，24H1 公司代销权益公募基金、非货币市场公募基金保有规模在券商中均位列第三，市占率总体稳定。  

        往后看，倘若基金投顾业务试点转常规，广发证券有望凭借强大的投研能力、完备的投顾业务服务体系实现基金投顾业务的“弯道超车”、巩固在财富管理业务的竞争优势。  

        ![](images/cbe364d93ce5f5d6a030cd82add102cedeaca3e3a568f8d5f7aa91864f9d9667.jpg)  
        图 5：广发证券代理交易金额及市占率  
        资料来源：公司公告、招商证券  

        ![](images/3c5ef0c36f1d500660bfbd6843afe88a5650fbb4a1aa5581b97bf813d1c301ee.jpg)  
        图 6：广发证券代销非货保有规模及市占率  
        资料来源：Wind、招商证券  

        # 3、交易机构业务相对稳健，衍生品聚焦做市  

        交易及机构业务相对稳健。广发证券交易及机构业务涵盖权益投资交易业务、固定收益销售及交易业务、股权衍生品销售及交易业务、另类投资业务、投资研究业务及资产托管业务。近年来，公司在权益类投资上坚持价值投资、严控仓位，在固收自营上控制债券投资组合的久期、杠杆和投资规模，在衍生品业务上持续为机构客户提供相关资产配置和风险管理解决方案，在投研业务上始终保持业内领先水平、屡获殊荣，综合努力下、交易及机构业务保持相对稳健、收入贡献水平逐年走高。24H1广发证券交易及机构业务收入达27.8亿元,同比增长 $13.5\%$ ，收入贡献达 $23.6\%$ ，同比上行 5.1pct。  

        ![](images/6084592059597b3b3e7a26dec07ee82f7698416ac04eb6e9fc3a744af05112aa.jpg)  
        图 7：广发证券交易及机构业务收入及收入贡献  
        资料来源：公司公告、招商证券  

        做市业务或带来一定业绩增量。目前，在监管压降业务杠杆、产品创新受限、券商外部资本补充渠道收窄的环境下，做市业务作为同时兼顾“功能性”作用发挥和“盈利性”的衍生品业务有望持续闪光。而广发证券做市业务长期保持在市场第一梯队，为上交所、深交所的 700 多只基金、全部 ETF 期权、中金所的沪深300/中证 1000 股指期权提供做市服务。往后看，伴随 ETF、上市证券/债券、股指期权等可做市产品增加，广发证券有望优先受益。  

        # 4、投行业绩爬坡加速  

        公司投行业务爬坡加速。2020 年受康美药业造假案影响，广发证券被暂停保荐资格 6 个月、债券承销业务受限 12 个月，直至 2021 年公司保荐资格和债券承销业务先后恢复、重获券商分类评价 AA 评级，公司投行风险才画上句号、业务进入复苏阶段。2024 年广发证券债券总承销金额 3374 亿，行业排名稳定在第14 位；IPO 募集资金 8.05 亿，行业排名从第 27 位上升至第 18 位。  

        ![](images/3f1cbde57ea90a0db82994b35b502c62c5ba5c9a127256011fd34e11c9b456d9.jpg)  
        图 8：IPO 募集资金排名：广发证券 vs 头部同业  
        资料来源：Wind、招商证券  

        ![](images/3b21ef981d15c58bec7c870d395ef15f7445bf12ee26355a4809799fc8986d29.jpg)  
        图 9：债券承销金额排名：广发证券 vs 头部同业  
        资料来源：Wind、招商证券  

        外部总量控制、内部基数效应加持，投行收入增幅领先同业。而在一二级市场再平衡的政策背景下，公司投行业务的相对低基数为高业绩弹性释放提供基础。  

        24Q3 广发证券投行净收入 5.3 亿，同比增长 $33\%$ ，增速领跑头部券商。  

        ![](images/5cd7006339152e24e9976c74e4d8ac09c109a2376adcb1ee86eb1718b4f07691.jpg)  
        图 10：24Q3 投行手续费净收 $\mathrm{\land}$ 及同比：广发证券 vs 头部同业  

        资料来源：Wind、招商证券  

        # 5、增资香港子公司，发力海外业务  

        三度增资香港子公司，深化国际业务布局。广发证券于上世纪 90 年代初期便旗帜鲜明地提出“股份化、集团化、国际化、规范化”的“四化”发展战略。进入2024 年后，公司国际化布局全面提速，广发控股（香港）对惠理集团 $20.04\%$ 的股权收购完成交割、补强公司境外资产管理业务竞争力；2024 年 5 月、7 月和2025 年 1 月公司分别向广发控股（香港）增资 15 亿港元、11 亿港元和 21.37亿港元，增资后广发香港（控股）的实缴资本增加至 103.37 亿港元、资本实力的增强或有助于广发打开海外业务的发展空间。  

        从业绩层面来看，香港市场整体回暖提振广发控股（香港）营收表现。24H1 广发控股（香港）实现营业收入 4.66 亿、净利润达 1.64 亿。  

        ![](images/3a07e8f9bd06ef95d19a7be6efae20b0606def17dc13ac46dc329171683edcf7.jpg)  
        图 11：广发香港（控股）营业总收入及净利润  
        资料来源：公司公告、招商证券  

        # 二、稳定的股权结构和经验丰富的管理层为公司穿  

        # 越周期提供坚实保障  

        股权结构整体稳定。吉林敖东、辽宁成大和中山公用 25 年来一直位列前三大股东，股东结构的长期稳定是公司具备高度凝聚力和战斗力、不断穿越周期、突破发展瓶颈、奠定行业地位的重要支撑。2 月 12 日晚间，公司第二大股东辽宁成大公告，授权管理层处置所持有的不超过广发证券总股本 $3\%$ 的 A 股股份。在券商并购重组市场环境和广东省政府印发《关于高质量发展资本市场助力广东现代化建设的若干措施》1政策背景下，此授权事项引发了市场对广发证券股权变更的遐想。但目前看，授权事项并非确定性执行方案，仍待进一步观察。  

        ![](images/02148059b79bab0e6dc847df11c1257360e960b0aa1a191343e506eae06cf02e.jpg)  
        图 12：广发证券股权结构  
        资料来源：公司公告、招商证券  

        管理层具备丰富的业务及管理经验。广发证券现任董事长林传辉先生自 1995 年12 月加入公司、历任投行部常务副总经理、广发基金总经理和副董事长等核心部门领导岗位，业务管理经验丰富、高度认同公司企业文化。截至目前，公司经营管理团队在证券、金融和经济相关领域的经历平均约 28 年，在公司平均任职期限超过 18 年，团队战略视野前瞻、专业实力过硬、文化归属感强烈、持续为公司高质量发展保驾护航。  

        表 1：广发证券管理层主要人员及工作经历  


        <html><body><table><tr><td>姓名</td><td>职务</td><td>工作履历</td></tr><tr><td>林传辉</td><td>董事长</td><td>自2020年12月起获委任为公司总经理，自2021年1月起获委任为公司执行董事，自2021 年7月起获委任为公司董事长。历任公司上海业务总部总经理、投资银行部常务副总经理、 广发基金总经理/副董事长、广发控股（香港）董事长。</td></tr><tr><td>秦力</td><td>执行董事、总经理</td><td>自2011年4月起获委任为公司执行董事，2020年12月起获委任为公司总监，2024年5 月起获委任为公司总经理。历任资金营运部总经理、公司总经理助理、易方达基金董事、 广发资管董事长、广发控股（香港）董事长。</td></tr><tr><td>孙晓燕</td><td>执行董事、常务副总经 理兼财务总监</td><td>12月获委任为公司执行董事。历任财务部总经理、广发基金财务总监/副总经理、证通公司 监事会主席。</td></tr><tr><td>肖雪生</td><td>执行董事、副总经理</td><td>自2010年7月起任广发信德董事，自2021年9月起任广发信德董事长，自2024年5月 获委任为公司执行董事、副总经理。历任投行业务管理总部副总经理、广发信德总经理。</td></tr></table></body></html>  

        <html><body><table><tr><td>欧阳西</td><td>副总经理、公司总监</td><td>自2020年12月起获委任为公司总监，自2024年5月获委任公司副总经理。历任投资银 行部副总经理及常务副总经理、投资自营部总经理、董事会秘书、财务总监、广发合信产 业投资管理有限公司董事长、广州投资顾问学院管理有限公司董事。</td></tr><tr><td>张威</td><td>副总经理</td><td>自2014年5月起获委任为本公司副总经理。历任固定收益总部总经理、公司总经理助理、 广发资管董事长、广发合信产业投资管理有限公司董事长、广发控股（香港）董事、广发 融资租赁董事长。</td></tr><tr><td>易阳方</td><td>副总经理</td><td>自2021年7月起获委任为本公司副总经理。历任公司投资银行总部、投资理财总部、投 资自营部业务员、副经理、广发基金常务副总经理、易方达基金董事。</td></tr><tr><td>辛治运</td><td>副总经理兼首席信息官</td><td>自2019年5月起获委任为公司首席信息官，自2021年7月起获委任为公司副总经理。历 任公司首席风险官、广发控股（香港）董事。</td></tr><tr><td>李谦</td><td>副总经理</td><td>自2021年7月起获委任为公司副总经理。历任公司固定收益销售交易部总经理、公司总 经理助理、证券投资业务管理总部总经理。</td></tr><tr><td>徐佑军</td><td>副总经理</td><td>自2019年4月起获委任为公司董事会秘书、联席公司秘书，自2021年7月起获委任为公 司副总经理、合规总监。历任公司投资银行部经理、董事会办公室总经理、公司证券事务 代表。</td></tr><tr><td>胡金泉</td><td>副总经理</td><td>自2021年8月起获委任为公司总经理助理兼任投行业务管理委员会副主任委员。历任公 司投资银行部总监、董事总经理。</td></tr><tr><td>吴顺虎</td><td>合规总监</td><td>自2022年1月起获委任为公司首席风险官。历任广发资管副总经理/首席风险官/合规负责 人、公司合规与法律事务部总经理。</td></tr><tr><td>崔舟航</td><td>首席风险官</td><td>历任广发控股（香港）首席风险官、公司人力资源管理部总经理。</td></tr><tr><td>尹中兴</td><td>董事会秘书</td><td>自2022年9月起任本公司董事会办公室总经理，自2024年5月起获委任为公司董事会秘 书。历任公司战略发展部执行董事。</td></tr></table></body></html>

        资料来源：Wind、招商证券  

        # 三、投资建议与风险提示  

        # 1、行业景气度及盈利预测  

        监管延续呵护市场。决策层从 2023 年 7 月 24 日中央政治局会议提出“活跃资本市场、提振投资者信心”、到今年 9 月 24 日金融支持经济高质量发展新闻发布会打出两项创新货币工具、降准降息等政策“组合拳”、9 月 26 日中央政治局会议强调“努力提振资本市场”，再到 12 月 9 日中央政治局会议明确“稳住楼市股市”，监管层从“两强两严”“长牙带刺”、到“突出维护市场稳定”这个关键，自上而下、宽严相济，推动权益市场走出一段温和、健康的上涨行情。  

        流动性持续宽松。宏观层面，12 月 9 日中央政治局会议提到“实施更加积极的财政政策和适度宽松的货币政策”、“加强超常规逆周期调节”，打开市场对于货币政策的想象空间，宏观流动性充盈。中观层面，“统筹推动融资端和投资端改革，逐步实现 IPO 常态化”稳定市场资金需求处于合理区间，两批 SFISF 合计1050 亿落地、工具弹性空间仍巨大，险资持续壮大、加码权益市场，股票增持回购再贷款提振公司及主要股东增持回购能力、进一步盘活市场，市场科技投资主线明晰带动资产重估、推动交投情绪再上台阶。总体看，市场资金面供大于求、  

        # 整体宽松。  

        基于政策展望、流动性展望，我们作出如下假设：假设 2025 年日均股基成交额为 1.45 万亿，两融余额均值为 1.72 万亿，股票表内质押规模达 1495 亿；IPO规模 1121 亿，再融资规模 5351 亿，债承规模 18.6 万亿；资管规模为 10.6 万亿；投资业务规模为 6.7 万亿。  

        表 2：证券行业景气度预测  


        <html><body><table><tr><td></td><td>亿元</td><td>2022</td><td>2023</td><td>2024E</td><td>2025E</td><td>2026E</td></tr><tr><td></td><td>三大指数算数平均涨跌幅</td><td>-22%</td><td>-11%</td><td>15%</td><td>10%</td><td>8%</td></tr><tr><td>经纪</td><td>股基交易额（双）</td><td>4,953,419</td><td>4,799,856</td><td>5,857,306</td><td>6,785,083</td><td>7,394,102</td></tr><tr><td></td><td>日均股基交易额</td><td>10,234</td><td>9,917</td><td>12,102</td><td>14,522</td><td>15,684</td></tr><tr><td>信用</td><td>两融余额均值</td><td>16,185</td><td>16,058</td><td>15,671</td><td>17,238</td><td>18,100</td></tr><tr><td></td><td>股票质押表内规模</td><td>2,124</td><td>1,832</td><td>1,514</td><td>1,495</td><td>1,569</td></tr><tr><td>投行</td><td>上市公司家数</td><td>5,079</td><td>5,346</td><td>5,448</td><td>5,598</td><td>5,748</td></tr><tr><td></td><td>IPO规模</td><td>5,869</td><td>3,565</td><td>663</td><td>1,121</td><td>1,121</td></tr><tr><td></td><td>再融资规模</td><td>7,844</td><td>5,940</td><td>2,007</td><td>5,351</td><td>5,494</td></tr><tr><td></td><td>债承规模</td><td>104,555</td><td>135,045</td><td>141,382</td><td>186,444</td><td>202,811</td></tr><tr><td>自营</td><td>自营规模</td><td>52,300</td><td>60,145</td><td>63,754</td><td>66,941</td><td>70,288</td></tr><tr><td>资管</td><td>资管规模</td><td>97,600</td><td>88,300</td><td>100,676</td><td>106,009</td><td>109,614</td></tr></table></body></html>

        资料来源：Wind、招商证券  

        # 2、公司经营业绩展望及盈利预测  

        基于当前行业发展趋势、景气度假设和公司战略规划，结合前述分部业务的景气度预测，我们对广发证券 2024-2026 年业绩进行预测。  

        表 3：广发证券分业务预测  


        <html><body><table><tr><td>单位：亿元</td><td></td><td>2022</td><td>2023</td><td>2024E</td><td>2025E</td><td>2026E</td></tr><tr><td>一、经纪</td><td>股基市占率</td><td>4.02%</td><td>3.87%</td><td>4.06%</td><td>4.14%</td><td>4.23%</td></tr><tr><td></td><td>股基交易额（双）</td><td>199,029</td><td>185,670</td><td>237,904</td><td>281,099</td><td>312,456</td></tr><tr><td></td><td>净佣金率</td><td>0.02%</td><td>0.02%</td><td>0.02%</td><td>0.02%</td><td>0.02%</td></tr><tr><td></td><td>经纪业务净收入</td><td>64</td><td>58</td><td>62</td><td>72</td><td>81</td></tr><tr><td>二、投行</td><td>IPO规模</td><td>29</td><td>20</td><td>8</td><td>16</td><td>22</td></tr><tr><td></td><td>再融资规模</td><td>155</td><td>143</td><td>99</td><td>193</td><td>209</td></tr><tr><td></td><td>债承规模</td><td>1,421</td><td>2,444</td><td>3,374</td><td>3,711</td><td>4,083</td></tr><tr><td></td><td>投行业务净收入</td><td>6.1</td><td>5.7</td><td>6.1</td><td>7.4</td><td>8.8</td></tr><tr><td>三、资管</td><td>资管规模</td><td>2,712</td><td>2,047</td><td>2,597</td><td>3,180</td><td>3,746</td></tr><tr><td></td><td>资管净收入</td><td>89</td><td>77</td><td>70</td><td>76</td><td>80</td></tr><tr><td>四、自营</td><td>交易性金融资产</td><td>1,578</td><td>2,161</td><td>2,727</td><td>2,904</td><td>3,148</td></tr><tr><td></td><td>其他债权投资</td><td>1,439</td><td>1,393</td><td>1,407</td><td>1,421</td><td>1,435</td></tr><tr><td></td><td>投资收益率</td><td>0.90%</td><td>1.91%</td><td>2.04%</td><td>2.10%</td><td>2.12%</td></tr><tr><td></td><td>自营净收入</td><td>13</td><td>36</td><td>70</td><td>82</td><td>87</td></tr><tr><td>五、信用</td><td>两融市占率</td><td>5.39%</td><td>5.39%</td><td>5.39%</td><td>5.71%</td><td>5.83%</td></tr><tr><td></td><td>两融年末余额</td><td>830</td><td>890</td><td>1,005</td><td>1,028</td><td>1,049</td></tr><tr><td></td><td>买入返售金融资产</td><td>189</td><td>197</td><td>228</td><td>260</td><td>288</td></tr><tr><td></td><td>信用业务净收入</td><td>41</td><td>31</td><td>27</td><td>36</td><td>46</td></tr></table></body></html>

        资料来源：Wind、招商证券  

        基于以上景气度预测，预计 24/25/26 年广发证券营业收入分别为 270/311/342亿，同比 $+16\%/{+}15\%/{+}10\%$ ；归母净利润分别为 82/101/118 亿，同比$+18\%/{+24\%}/{+16\%}$ 。24/25/26 年 BPS 为 16.1/17.5/19.2 元，现价对应 PB 估值分别为 $0.97/0.89/0.81$ 倍。  

        # 3、投资建议  

        维持“强烈推荐”评级。广发证券整体稳定的股权结构、成长于一线、经验丰富的经营管理团队保证了公司的战略执行力、团队凝聚力。业务层面，广发证券“大资管”业务领先，控股子公司广发基金、参股公司易方达基金在行业内头部“交椅”稳固，尤其在被动化投资趋势下易方达凭借声誉优势、规模优势和投研实力有望持续扩宽、加深公司在资管领域的竞争优势；财富管理业务持续闪亮，金融科技赋能直指投资者交易痛点、堵点，传统经纪基本盘稳固；公司战略性打造、培育专业投顾团队，加之广州投顾业务生态建设持续完善，强劲的内核力量和良好的外部环境有望加速财富管理转型升级；交易及机构业务稳健，公司聚焦做市业务、很好理顺了“功能性”和“盈利性”之间的关系，投研长期保持行业前列、屡获殊荣；投行业务发展进入稳定期、收入加速爬坡中；国际化布局持续深入。  

        展望后市，资本市场仍处于监管呵护的大周期内，而伴随新一轮科技投资浪潮兴起，中国资产价值迎来重估，券商并购重组“下半场”开启在即，市场风偏持续向好，流动性充裕、赚钱效应充足，券商业绩、估值均有望改善。对比头部同业，目前广发证券 PB 估值相对较低，与公司潜在的业绩改善空间不匹配，整体安全边际较高。我们预计 24/25/26 年广发证券归母净利润分别为 82/101/118 亿，同比 $+18\%/+24\%/+16\%$ 。给定公司目标价 19.3 元，对应 1.1 倍 PB 目标，空间约为 $24\%$ ，维持强烈推荐评级。  

        # 4、风险提示  

        政策力度或有效性不及预期：政策对市场的实际刺激效果难以预测；  

        市场萎靡或投资情绪低迷：若市场情绪维持低迷，可能会影响公司投资收益和股价走向；  

        业务费率下行拖累收入下滑：如果交易佣金和基金管理费等降低力度超预期，则在一定程度上拖累公司手续费收入和基金业务收入。  

        公司市占率提升不及预期：如果市场竞争加剧，公司各项分部业务市占率提升可能会不及预期；  

        合规风险：在监管趋严的背景下，公司合规风险或将拖累业务发展。  

        ![](images/5b36cd7f3ebbca427de1a63990039bf103d4bc8330f850505049247060f16fc0.jpg)  
        图 13：广发证券历史 PE Band  
        资料来源：公司数据、招商证券  

        ![](images/42d50d551d2b8c06e0e1c16262a88949da3fbcf64ca91e73f7ed601c64b505cc.jpg)  
        图 14：广发证券历史 PB Band  

        资料来源：公司数据、招商证券  

        # 附：财务预测表  

        表 4：损益表 （单位：百万元）  


        <html><body><table><tr><td></td><td>2021</td><td>2022</td><td>2023</td><td>2024E</td><td>2025E</td><td>2026E</td></tr><tr><td>营业收入</td><td>34,250</td><td>25,132</td><td>23,300</td><td>27,016</td><td>31,093</td><td>34,248</td></tr><tr><td>手续费及佣金净收入</td><td>18,785</td><td>16,363</td><td>14,512</td><td>14,250</td><td>15,994</td><td>17,475</td></tr><tr><td>代理买卖证券业务净收入</td><td>7,970</td><td>6,387</td><td>5,810</td><td>6,248</td><td>7,230</td><td>8,110</td></tr><tr><td>证券承销业务净收入</td><td>433</td><td>610</td><td>566</td><td>606</td><td>741</td><td>883</td></tr><tr><td>受托客户资产管理业务净收入</td><td>9,946</td><td>8,939</td><td>7,728</td><td>6,955</td><td>7,561</td><td>7,999</td></tr><tr><td>利息净收入</td><td>4,931</td><td>4,101</td><td>3,136</td><td>2,657</td><td>3,641</td><td>4,553</td></tr><tr><td>投资净收益</td><td>6,817</td><td>4,383</td><td>5,301</td><td>7,571</td><td>8,231</td><td>8,803</td></tr><tr><td>其中：对联营企业和合营企业的投资</td><td>1,464</td><td>935</td><td>723</td><td>1,514</td><td>1,646</td><td>1,761</td></tr><tr><td>公允价值变动净收益</td><td>407</td><td>-2,183</td><td>-1,011</td><td>891</td><td>1,453</td><td>1,553</td></tr><tr><td>汇兑净收益</td><td>4</td><td>-47</td><td>-10</td><td>10</td><td>10</td><td>10</td></tr><tr><td>其他收益</td><td>1,228</td><td>1,433</td><td>982</td><td>903</td><td>994</td><td>1,043</td></tr><tr><td>其他业务收入</td><td>2,075</td><td>1,082</td><td>388</td><td>733</td><td>770</td><td>810</td></tr><tr><td>资产处置收益</td><td>2</td><td>0</td><td>1</td><td></td><td></td><td></td></tr><tr><td>营业支出</td><td>19,225</td><td>14,684</td><td>14,505</td><td>16,430</td><td>18,161</td><td>19,151</td></tr><tr><td>税金及附加</td><td>223</td><td>175</td><td>166</td><td>270</td><td>311</td><td>342</td></tr><tr><td>管理费用</td><td>15,961</td><td>13,809</td><td>13,885</td><td>15,172</td><td>16,882</td><td>17,726</td></tr><tr><td>资产减值损失</td><td>3</td><td>12</td><td>4</td><td>135</td><td>58</td><td>118</td></tr><tr><td>信用减值损失</td><td>981</td><td>-372</td><td>95</td><td>135</td><td>155</td><td>171</td></tr><tr><td>其他业务成本</td><td>2,058</td><td>1,060</td><td>355</td><td>719</td><td>755</td><td>794</td></tr><tr><td>营业利润</td><td>15,025</td><td>10,448</td><td>8,795</td><td>10,586</td><td>12,932</td><td>15,097</td></tr><tr><td>加：营业外收入</td><td>8</td><td>3</td><td>1</td><td>1</td><td>2</td><td>2</td></tr><tr><td>减：营业外支出</td><td>69</td><td>63</td><td>52</td><td>60</td><td>69</td><td>76</td></tr><tr><td>其中：非流动资产处置净损失</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>利润总额</td><td>14,964</td><td>10,388</td><td>8,744</td><td>10,527</td><td>12,864</td><td>15,023</td></tr><tr><td>减：所得税</td><td>2,909</td><td>1,490</td><td>882</td><td>1,211</td><td>1,479</td><td>1,728</td></tr><tr><td>加：未确认的投资损失</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>净利润</td><td>12,055</td><td>8,898</td><td>7,863</td><td>9,317</td><td>11,385</td><td>13,295</td></tr><tr><td>减：少数股东损益</td><td>1,201</td><td>969</td><td>885</td><td>1,118</td><td>1,255</td><td>1,501</td></tr><tr><td>归属于母公司所有者的净利润</td><td>10,854</td><td>7,929</td><td>6,978</td><td>8,199</td><td>10,130</td><td>11,794</td></tr><tr><td>加：其他综合收益</td><td>-54</td><td>-310</td><td>602</td><td>4,147</td><td>4,008</td><td>4,199</td></tr><tr><td>综合收益总额</td><td>12,001</td><td>8,588</td><td>8,465</td><td>13,464</td><td>15,393</td><td>17,494</td></tr><tr><td>减：归属于少数股东的综合收益总额</td><td>1,194</td><td>983</td><td>888</td><td>1,481</td><td>1,650</td><td>1,909</td></tr><tr><td>归属于母公司普通股东综合收益总额</td><td>10,807</td><td>7,605</td><td>7,577</td><td>11,983</td><td>13,743</td><td>15,585</td></tr><tr><td>每股收益：</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>基本每股收益 （元） 稀释每股收益 （元）)</td><td>1.42 1.42</td><td>1.02 1.02</td><td>0.83 0.83</td><td>1.08 1.08</td><td>1.33</td><td>1.55</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>1.33</td><td>1.55</td></tr></table></body></html>

        资料来源：公司数据、招商证券  

        表 5：资产负债表（单位：百万元）  


        <html><body><table><tr><td></td><td>2021</td><td>2022</td><td>2023</td><td>2024E</td><td>2025E</td><td>2026E</td></tr><tr><td>资产：</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>货币资金</td><td>119,313</td><td>129,176</td><td>118,815</td><td>162,523</td><td>185,511</td><td>205,703</td></tr><tr><td>其中：客户资金存款</td><td>97,497</td><td>107,607</td><td>94,839</td><td>133,215</td><td>152,058</td><td>168,609</td></tr><tr><td>结算备付金</td><td>27,694</td><td>27,680</td><td>34,510</td><td>35,343</td><td>44,280</td><td>49,405</td></tr><tr><td>其中：客户备付金</td><td>23,147</td><td>23,398</td><td>29,648</td><td>30,042</td><td>37,638</td><td>41,994</td></tr><tr><td>拆出资金</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>融出资金</td><td>97,231</td><td>82,823</td><td>91,108</td><td>96,487</td><td>98,514</td><td>98,481</td></tr><tr><td>金融投资</td><td>235,925</td><td>302,820</td><td>361,196</td><td>427,239</td><td>447,401</td><td>474,439</td></tr><tr><td>其中：交易性金融资产</td><td>124,473</td><td>157,801</td><td>216,074</td><td>272,738</td><td>290,397</td><td>314,830</td></tr><tr><td>债权投资</td><td>105</td><td>354</td><td>130</td><td>140</td><td>143</td><td>146</td></tr><tr><td>其它债权投资</td><td>110,475</td><td>143,938</td><td>139,295</td><td>140,688</td><td>142,095</td><td>143,516</td></tr><tr><td>其他权益工具投资</td><td>873</td><td>728</td><td>5,697</td><td>13,673</td><td>14,766</td><td>15,948</td></tr><tr><td>以摊余成本计量的金融资产 衍生金融资产</td><td>564</td><td>2,642</td><td>5,034</td><td>11,455</td><td>12,197</td><td>13,223</td></tr><tr><td>买入返售金融资产 持有待售资产</td><td>19,992</td><td>18,940</td><td>19,721</td><td>22,753</td><td>25,972</td><td>28,798</td></tr><tr><td>应收款项 合同资产</td><td>4,893</td><td>13,772</td><td>11,149</td><td>7,273</td><td>9,328</td><td>10,274</td></tr><tr><td>应收利息 存出保证金</td><td>12,495</td><td>20,342</td><td></td><td></td><td></td><td></td></tr><tr><td>代理业务资产 可供出售金融资产</td><td></td><td></td><td>21,253</td><td>18,086</td><td>24,573</td><td>29,505</td></tr><tr><td>持有至到期投资</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>长期股权投资</td><td>8,248</td><td>8,744</td><td>9,225</td><td>9,687</td><td>10,171</td><td>10,679</td></tr><tr><td>固定资产</td><td>2,967</td><td>2,833</td><td>2,847</td><td>2,746</td><td>2,677</td><td>2,595</td></tr><tr><td>在建工程</td><td></td><td>246</td><td>246</td><td>259</td><td>272</td><td>285</td></tr><tr><td>无形资产</td><td>1,490</td><td>1,546</td><td>1,597</td><td></td><td></td><td></td></tr><tr><td>其中：交易席位费</td><td></td><td></td><td></td><td>1,677</td><td>1,761</td><td>1,849</td></tr><tr><td>商誉</td><td>2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>递延所得税资产</td><td>2,119</td><td>2</td><td>2</td><td>2</td><td>2</td><td>3</td></tr><tr><td>投资性房地产</td><td>61</td><td>2,583</td><td>2,562</td><td>1,711</td><td>2,720</td><td>3,439</td></tr><tr><td>使用权资产</td><td></td><td>187</td><td>199</td><td>219</td><td>241</td><td>265</td></tr><tr><td></td><td>818</td><td>765</td><td>948</td><td>948</td><td>948</td><td>948</td></tr><tr><td>其他资产</td><td>2,043</td><td>2,152</td><td>1,768</td><td>1,800</td><td>1,800</td><td>1,800</td></tr><tr><td>资产总计</td><td>535,855</td><td>617,256</td><td>682,182</td><td>713,663</td><td>792,376</td><td>881,684</td></tr><tr><td>负债：</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>短期借款</td><td>917</td><td>4,492</td><td>6,838</td><td>8,497</td><td>6,609</td><td>7,553</td></tr><tr><td>其中：质押借款</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>应付短期融资款</td><td>27,877</td><td>37,308</td><td>45,363</td><td>58,972</td><td>82,561</td><td>115,586</td></tr><tr><td>拆入资金</td><td>11,617</td><td>19,071</td><td>22,653</td><td>28,316</td><td>32,564</td><td>37,448</td></tr><tr><td>交易性金融负债</td><td>10,823</td><td>11,985</td><td>17,609</td><td>21,131</td><td>25,357</td><td>32,964</td></tr><tr><td>衍生金融负债</td><td>981</td><td>2,098</td><td>4,701</td><td>1,057</td><td>1,268</td><td>1,648</td></tr><tr><td>卖出回购金融资产款</td><td>81,230</td><td>125,058</td><td>153,749</td><td>110,515</td><td>126,148</td><td>139,878</td></tr><tr><td>代理买卖证券款</td><td>126,731</td><td>137,585</td><td>132,011</td><td>171,291</td><td>199,580</td><td>221,844</td></tr><tr><td>代理承销证券款</td><td></td><td>149</td><td></td><td>0</td><td>75</td><td>75</td></tr><tr><td>应付职工薪酬</td><td>10,118</td><td>10,147</td><td>9,496</td><td>9,103</td><td>9,791</td><td>10,281</td></tr><tr><td>长期应付职工薪酬</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>应交税费</td><td>1,645</td><td>900</td><td>556</td><td>912</td><td>1,115</td><td>1,302</td></tr><tr><td>应付款项</td><td>7,074</td><td>21,809</td><td>37,138</td><td>48,280</td><td>47,314</td><td>46,368</td></tr><tr><td>应付利息</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>合同负债 持有待售负债</td><td>111</td><td>440</td><td>116</td><td>116</td><td>116</td><td>116</td></tr></table></body></html>  

        <html><body><table><tr><td></td><td>2021</td><td>2022</td><td>2023</td><td>2024E</td><td>2025E</td><td>2026E</td></tr><tr><td>代理业务负债</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>长期借款</td><td>339</td><td>65</td><td></td><td>0</td><td>0</td><td>0</td></tr><tr><td>应付债券</td><td>138,683</td><td>115,887</td><td>103,580</td><td>98,401</td><td>92,497</td><td>86,948</td></tr><tr><td>递延所得税负债</td><td>741</td><td>574</td><td>449</td><td>582</td><td>711</td><td>831</td></tr><tr><td>预计负债</td><td>406</td><td>440</td><td>447</td><td>447</td><td>447</td><td>447</td></tr><tr><td>租赁负债</td><td>842</td><td>789</td><td>970</td><td>970</td><td>970</td><td>970</td></tr><tr><td>其他负债</td><td>4,918</td><td>4,012</td><td>5,830</td><td>4,756</td><td>3,805</td><td>3,044</td></tr><tr><td>负债合计</td><td>425,054</td><td>492,463</td><td>541,506</td><td>563,347</td><td>630,928</td><td>707,302</td></tr><tr><td>所有者权益(或股东权益):</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>股本</td><td>7,621</td><td>7,621</td><td>7,621</td><td>7,621</td><td>7,606</td><td>7,606</td></tr><tr><td>其它权益工具</td><td>1,000</td><td>10,990</td><td>22,479</td><td>22,479</td><td>22,479</td><td>22,479</td></tr><tr><td>其中：优先股</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>永续债</td><td>1,000</td><td>10,990</td><td>22,479</td><td>22,479</td><td>22,479</td><td>22,479</td></tr><tr><td>资本公积金</td><td>31,284</td><td>31,286</td><td>31,297</td><td>31,297</td><td>31,297</td><td>31,297</td></tr><tr><td>减：库存股</td><td></td><td>234</td><td>234</td><td>234</td><td>218</td><td>218</td></tr><tr><td>其它综合收益</td><td>1,060</td><td>735</td><td>1,339</td><td>5,123</td><td>8,736</td><td>12,527</td></tr><tr><td>盈余公积金</td><td>7,948</td><td>8,733</td><td>9,431</td><td>10,363</td><td>11,501</td><td>12,831</td></tr><tr><td>未分配利润</td><td>38,140</td><td>39,266</td><td>40,149</td><td>42,613</td><td>46,490</td><td>51,205</td></tr><tr><td>一般风险准备</td><td>19,572</td><td>21,748</td><td>23,636</td><td>25,499</td><td>27,776</td><td>30,435</td></tr><tr><td>外币报表折算差额 未确认的投资损失</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>归属于母公司所有者权益合计</td><td>106,625</td><td>120,146</td><td>135,718</td><td>144,760</td><td>155,666</td><td>168,160</td></tr><tr><td>少数股东权益</td><td>4,177</td><td>4,647</td><td>4,958</td><td>5,556</td><td>5,781</td><td>6,222</td></tr><tr><td>所有者权益合计</td><td>110,801</td><td>124,793</td><td>140,676</td><td>150,316</td><td>161,448</td><td>174,382</td></tr><tr><td>负债及股东权益总计</td><td>535,855</td><td>617,256</td><td>682,182</td><td>713,663</td><td>792,376</td><td>881,684</td></tr></table></body></html>

        资料来源：公司数据、招商证券  

        # 分析师承诺  

        负责本研究报告的每一位证券分析师，在此申明，本报告清晰、准确地反映了分析师本人的研究观点。本人薪酬的任何部分过去不曾与、现在不与，未来也将不会与本报告中的具体推荐或观点直接或间接相关。  

        # 评级说明  

        报告中所涉及的投资评级采用相对评级体系，基于报告发布日后 6-12 个月内公司股价（或行业指数）相对同期当地市场基准指数的市场表现预期。其中，A 股市场以沪深 300 指数为基准；香港市场以恒生指数为基准；美国市场以标普 500 指数为基准。具体标准如下：  

        # 股票评级  

        强烈推荐：预期公司股价涨幅超越基准指数 $20\%$ 以上增持：预期公司股价涨幅超越基准指数 $5-20\%$ 之间中性：预期公司股价变动幅度相对基准指数介于 $\pm5\%$ 之间减持：预期公司股价表现弱于基准指数 $5\%$ 以上  

        # 行业评级  

        推荐：行业基本面向好，预期行业指数超越基准指数中性：行业基本面稳定，预期行业指数跟随基准指数回避：行业基本面转弱，预期行业指数弱于基准指数  

        # 重要声明  

        本报告由招商证券股份有限公司（以下简称“本公司”）编制。本公司具有中国证监会许可的证券投资咨询业务资格。本报告基于合法取得的信息，但本公司对这些信息的准确性和完整性不作任何保证。本报告所包含的分析基于各种假设，不同假设可能导致分析结果出现重大不同。报告中的内容和意见仅供参考，并不构成对所述证券买卖的出价，在任何情况下，本报告中的信息或所表述的意见并不构成对任何人的投资建议。除法律或规则规定必须承担的责任外，本公司及其雇员不对使用本报告及其内容所引发的任何直接或间接损失负任何责任。本公司或关联机构可能会持有报告中所提到的公司所发行的证券头寸并进行交易，还可能为这些公司提供或争取提供投资银行业务服务。客户应当考虑到本公司可能存在可能影响本报告客观性的利益冲突。  

        本报告版权归本公司所有。本公司保留所有权利。未经本公司事先书面许可，任何机构和个人均不得以任何形式翻版、复制、引用或转载，否则，本公司将保留随时追究其法律责任的权利。 """




    text2 = """# 股票名称(股票代码)  

子行业/行业  

发布时间：2025-04-30  

证券研究报告 / 公司点评报告  

# 高速通信线增速可喜，224G 产品已批量交付  

# ---沃尔核材 2024 年年度报告&2025 年一季报点评  

[事Ta件ble：_S公u司mm近ar日y]发布 2024 年年度报告和 2025 年第一季度报告，2024 年实现营业收入 69.27 亿元， $\mathrm{YoY+}21.03\%$ ；实现归母净利润 8.48 亿元，$\mathrm{YoY+}21.00\%$ ；毛利率 $31.73\%$ ，较去年同期降低 0.90pct。2025 年第一季度公司实现营业收入 17.59 亿元， $\mathrm{YoY+}26.60\%$ ；实现归母净利润 2.50 亿元， $\mathrm{YoY+35.86\%}$ ；毛利率 $32.53\%$ ，较去年同期降低0.63pct。整体业绩保持快速增长。  

点评：高速通信线缆业务快速增长，降本增效增厚利润。分产品来看，2024 年公司电子材料业务实现收入 25.99 亿元， $\mathrm{YoY+l8.25\%}$ ，毛利率$40.52\%$ ，较上年提升 3.60pct；通信线缆业务实现收入 17.02 亿元，$\mathrm{YoY+46.18\%}$ ，毛利率 $17.11\%$ ，较上年提升 $1.27\mathrm{pct}$ ；电力产品业务实现收入 9.27 亿元， $Y_{0}\mathrm{Y}{-}2.78\%$ ，毛利率 $38.11\%$ ，较上年降低 $4.37\mathrm{pct}$ ；新能源汽车产品业务实现收入13.81 亿元， $\mathrm{YoY+}27.62\%$ ，毛利率 $24.45\%$ ，较上年降低 3.06pct；风力发电业务实现收入1.52 亿元， $\mathrm{YoY-}4.40\%$ ，毛利率 $67.56\%$ ，较上年降低1.96pct；其他业务收入1.65 亿元， $Y_{0}\mathrm{Y}{-}0.62\%$ ，毛利率 $36.25\%$ ，较上年提升 1.02pct。通信线缆业务增速最快，2025 年一季度公司收入增长 $26.60\%$ ，主要来源于电子材料、通信线缆和新能源汽车产品的收入增长，其中高速通信线产品增速较快。费用端，2024 年公司销售/ 管理/ 研发费用为 3.54/3.02/3.49 亿元，分别同比增加$6.83\%/14.57\%/12.50\%$ ，在精益运营持续推进降本增效。  

224G 高速通信线批量交付中，前瞻性产能储备支撑未来长期增长。AI产业快速发展背景下高密度集成算力机柜对于高速高可靠通信线缆的需求持续攀升。据 Light Counting 预测，2023 年至 2027 年全球高速铜缆市场年复合增长率 $25\%$ ，到2027 年出货量预计达到2000 万条。公司通信线缆业务具备较强竞争优势，子公司乐庭智联凭借单通道 224G 高速通信线技术上的领先优势和设备及工艺壁垒率先抢占市场，服务于下游国内外龙头企业，224G 高速通信线产品正批量交付中，且目前已拥有绕包机近四百台，芯线机超三十台，此外已下单超两百台绕包机和几十台芯线机。新采购设备的陆续交付和产能提升将支撑订单收入规模持续增长。盈利预测：沃尔核材是国际领先的高速通信线供应商，看好公司凭借技术和产能储备优势保持竞争优势，实现业绩快速增长。预计公司 2025-27年实现营收 85.58/112.31/140.45 亿元，实现归母净利润 12.40/16.91/22.41亿元，对应 EPS 0.98/1.34/1.78 元，维持“买入”评级。  

# 风险提示：行业竞争加剧、下游需求不及预期。  

<html><body><table><tr><td>财务摘要 (百万元)</td><td>2023A</td><td>2024A</td><td>2025E</td><td>2026E</td><td>2027E</td></tr><tr><td>营业收入</td><td>5,723</td><td>6,927</td><td>8,558</td><td>11,231</td><td>14,045</td></tr><tr><td>(+/-)%</td><td>7.16%</td><td>21.03%</td><td>23.56%</td><td>31.23%</td><td>25.05%</td></tr><tr><td>归属母公司净利润</td><td>700</td><td>848</td><td>1,240</td><td>1,691</td><td>2,241</td></tr><tr><td>(+/-)%</td><td>13.97%</td><td>21.00%</td><td>46.28%</td><td>36.35%</td><td>32.54%</td></tr><tr><td>每股收益 (元)</td><td>0.56</td><td>0.68</td><td>0.98</td><td>1.34</td><td>1.78</td></tr><tr><td>市盈率</td><td>13.48</td><td>37.13</td><td>18.27</td><td>13.40</td><td>10.11</td></tr><tr><td>市净率</td><td>1.94</td><td>5.75</td><td>3.48</td><td>2.88</td><td>2.35</td></tr><tr><td>净资产收益率(%)</td><td>15.00%</td><td>16.29%</td><td>19.06%</td><td>21.52%</td><td>23.23%</td></tr><tr><td>股息收益率(%)</td><td>0.95%</td><td>0.76%</td><td>1.10%</td><td>1.50%</td><td>1.99%</td></tr><tr><td>总股本 (百万股)</td><td>1,260</td><td>1,260</td><td>1,260</td><td>1,260</td><td>1,260</td></tr></table></body></html>  

买入  

上次评级： 买入  

<html><body><table><tr><td>股票数据</td><td>2025/04/29</td></tr><tr><td>6个月目标价 (元) 收盘价 (元)</td><td></td></tr><tr><td>12个月股价区间（元）</td><td>17.98 10.73~29.22</td></tr><tr><td>总市值 (百万元)</td><td></td></tr><tr><td>总股本 (百万股)</td><td>22,652.98 1,260</td></tr><tr><td>A股 (百万股)</td><td>1,260</td></tr><tr><td>B股/H股(百万股)</td><td></td></tr><tr><td>日均成交量 (百万股)</td><td>0/0 48</td></tr></table></body></html>  

<html><body><table><tr><td>涨跌幅 (%)</td><td>1M</td><td>3M</td><td>12M</td></tr><tr><td>绝对收益</td><td>-11%</td><td>-30%</td><td>23%</td></tr><tr><td>相对收益</td><td>-8%</td><td>-29%</td><td>19%</td></tr></table></body></html>  

# [相Tab关le_报Re告po  

《沃尔核材（002130）：业绩实现快速增长，高速通信线缆业务有望加速放量》--20240821  

《沃尔核材（002130）：增资发泡材料公司深圳特发稳步推进，持续提升高速通信线缆业务竞争力》--20240802《沃尔核材（002130）：业绩实现快速增长，高速线缆业务有望加速放量》--20240710  

证券分析师：要文强  
执业证书编号：S055052301000413552769350 yao_wq@nesc.cn证券分析师：刘云坤  
执业证书编号：S055052405000115611880589 liuyk@nesc.cn  

附表：财务报表预测摘要及指标   


<html><body><table><tr><td>资产负债表 (百万元）</td><td>2024A</td><td>2025E</td><td>2026E</td><td>2027E</td></tr><tr><td>货币资金</td><td>1,028</td><td>1,442</td><td>2,150</td><td>3,321</td></tr><tr><td>交易性金融资产</td><td>145</td><td>145</td><td>145</td><td>145</td></tr><tr><td>应收款项</td><td>2.904</td><td>3,633</td><td>4,757</td><td>5,948</td></tr><tr><td>存货</td><td>865</td><td>1,082</td><td>1,409</td><td>1,749</td></tr><tr><td>其他流动资产</td><td>148</td><td>148</td><td>148</td><td>148</td></tr><tr><td>流动资产合计</td><td>5,538</td><td>6,985</td><td>9,312</td><td>12,192</td></tr><tr><td>可供出售金融资产</td><td></td><td></td><td></td><td></td></tr><tr><td>长期投资净额</td><td>57</td><td>57</td><td>57</td><td>57</td></tr><tr><td>固定资产</td><td>2,719</td><td>2,786</td><td>2,778</td><td>2,699</td></tr><tr><td>无形资产</td><td>289</td><td>269</td><td>249</td><td>229</td></tr><tr><td>商誉</td><td>695</td><td>695</td><td>695</td><td>695</td></tr><tr><td>非流动资产合计</td><td>4,728</td><td>4.876</td><td>4.837</td><td>4,728</td></tr><tr><td>资产总计</td><td>10,265</td><td>11,861</td><td>14,149</td><td>16,920</td></tr><tr><td>短期借款</td><td>344</td><td>344</td><td>344</td><td>344</td></tr><tr><td>应付款项</td><td>1,513</td><td>1,801</td><td>2,388</td><td>2,963</td></tr><tr><td>预收款项</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>一年内到期的非流动负债</td><td>186</td><td>281</td><td>281</td><td>281</td></tr><tr><td>流动负债合计</td><td>2,875</td><td>3,403</td><td>4,200</td><td>4,994</td></tr><tr><td>长期借款</td><td>901</td><td>901</td><td>901</td><td>901</td></tr><tr><td>其他长期负债</td><td>360</td><td>355</td><td>355</td><td>355</td></tr><tr><td>长期负债合计</td><td>1,262</td><td>1,257</td><td>1,257</td><td>1,257</td></tr><tr><td>负债合计</td><td>4,137</td><td>4,660</td><td>5,457</td><td>6,250</td></tr><tr><td>归属于母公司股东权益合计</td><td>5,535</td><td>6,505</td><td>7,855</td><td>9,644</td></tr><tr><td>少数股东权益</td><td>594</td><td>696</td><td>837</td><td>1,025</td></tr><tr><td>负债和股东权益总计</td><td>10,265</td><td>11,861</td><td>14,149</td><td>16,920</td></tr></table></body></html>  

<html><body><table><tr><td>利润表 (百万元)</td><td>2024A</td><td>2025E</td><td>2026E</td><td>2027E</td></tr><tr><td>营业收入</td><td>6,927</td><td>8,558</td><td>11,231</td><td>14,045</td></tr><tr><td>营业成本</td><td>4,729</td><td>5,850</td><td>7,653</td><td>9,497</td></tr><tr><td>营业税金及附加</td><td>57</td><td>73</td><td>95</td><td>119</td></tr><tr><td>资产减值损失</td><td>-69</td><td>0</td><td>0</td><td>0</td></tr><tr><td>销售费用</td><td>354</td><td>411</td><td>517</td><td>618</td></tr><tr><td>管理费用</td><td>302</td><td>359</td><td>449</td><td>534</td></tr><tr><td>财务费用</td><td>41</td><td>0</td><td>0</td><td>0</td></tr><tr><td>公允价值变动净收益</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>投资净收益</td><td>17</td><td>21</td><td>28</td><td>35</td></tr><tr><td>营业利润</td><td>1,089</td><td>1,544</td><td>2,118</td><td>2.807</td></tr><tr><td>营业外收支净额</td><td>-15</td><td>0</td><td>0</td><td>0</td></tr><tr><td>利润总额</td><td>1,074</td><td>1,544</td><td>2,118</td><td>2.807</td></tr><tr><td>所得税</td><td>153</td><td>202</td><td>286</td><td>379</td></tr><tr><td>净利润</td><td>921</td><td>1,342</td><td>1,832</td><td>2,428</td></tr><tr><td>归属于母公司净利润</td><td>848</td><td>1,240</td><td>1,691</td><td>2,241</td></tr><tr><td>少数股东损益</td><td>73</td><td>102</td><td>141</td><td>188</td></tr></table></body></html>  

<html><body><table><tr><td>现金流量表 (百万元）</td><td>2024A</td><td>2025E</td><td>2026E</td><td>2027E</td></tr><tr><td>净利润</td><td>921</td><td>1,342</td><td>1,832</td><td>2,428</td></tr><tr><td>资产减值准备</td><td>97</td><td>0</td><td>0</td><td>0</td></tr><tr><td>折旧及摊销</td><td>288</td><td>264</td><td>289</td><td>309</td></tr><tr><td>公允价值变动损失</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>财务费用</td><td>66</td><td>0</td><td>0</td><td>0</td></tr><tr><td>投资损失</td><td>-17</td><td>-21</td><td>-28</td><td>-35</td></tr><tr><td>运营资本变动</td><td>-467</td><td>-600</td><td>-822</td><td>-915</td></tr><tr><td>其他</td><td>11</td><td>-5</td><td>0</td><td>0</td></tr><tr><td>经营活动净现金流量</td><td>898</td><td>980</td><td>1,271</td><td>1,787</td></tr><tr><td>投资活动净现金流量</td><td>-545</td><td>-387</td><td>-222</td><td>-165</td></tr><tr><td>融资活动净现金流量</td><td>-330</td><td>-181</td><td>-341</td><td>-451</td></tr><tr><td>企业自由现金流</td><td>517</td><td>801</td><td>1,049</td><td>1,623</td></tr></table></body></html>  

<html><body><table><tr><td>财务与估值指标</td><td>2024A</td><td>2025E</td><td>2026E</td><td>2027E</td></tr><tr><td>每股指标</td><td></td><td></td><td></td><td></td></tr><tr><td>每股收益 (元)</td><td>0.68</td><td>0.98</td><td>1.34</td><td>1.78</td></tr><tr><td>每股净资产 (元)</td><td>4.39</td><td>5.16</td><td>6.23</td><td>7.65</td></tr><tr><td>每股经营性现金流量 (元)</td><td>0.71</td><td>0.78</td><td>1.01</td><td>1.42</td></tr><tr><td>成长性指标</td><td></td><td></td><td></td><td></td></tr><tr><td>营业收入增长率</td><td>21.0%</td><td>23.6%</td><td>31.2%</td><td>25.1%</td></tr><tr><td>净利润增长率</td><td>21.0%</td><td>46.3%</td><td>36.4%</td><td>32.5%</td></tr><tr><td>盈利能力指标</td><td></td><td></td><td></td><td></td></tr><tr><td>毛利率</td><td>31.7%</td><td>31.6%</td><td>31.9%</td><td>32.4%</td></tr><tr><td>净利润率</td><td>12.2%</td><td>14.5%</td><td>15.1%</td><td>16.0%</td></tr><tr><td>运营效率指标</td><td></td><td></td><td></td><td></td></tr><tr><td>应收账款周转天数</td><td>120.83</td><td>117.36</td><td>114.53</td><td>116.98</td></tr><tr><td>存货周转天数</td><td>59.97</td><td>59.90</td><td>58.58</td><td>59.86</td></tr><tr><td>偿债能力指标</td><td></td><td></td><td></td><td></td></tr><tr><td>资产负债率</td><td>40.3%</td><td>39.3%</td><td>38.6%</td><td>36.9%</td></tr><tr><td>流动比率</td><td>1.93</td><td>2.05</td><td>2.22</td><td>2.44</td></tr><tr><td>速动比率</td><td>1.54</td><td>1.66</td><td>1.81</td><td>2.02</td></tr><tr><td>费用率指标</td><td></td><td></td><td></td><td></td></tr><tr><td>销售费用率</td><td>5.1%</td><td>4.8%</td><td>4.6%</td><td>4.4%</td></tr><tr><td>管理费用率</td><td>4.4%</td><td>4.2%</td><td>4.0%</td><td>3.8%</td></tr><tr><td>财务费用率</td><td>0.6%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>分红指标</td><td></td><td></td><td></td><td></td></tr><tr><td>股息收益率</td><td>0.8%</td><td>1.1%</td><td>1.5%</td><td>2.0%</td></tr><tr><td>估值指标</td><td></td><td></td><td></td><td></td></tr><tr><td>P/E (倍)</td><td>37.13</td><td>18.27</td><td>13.40</td><td>10.11</td></tr><tr><td>P/B (倍)</td><td>5.75</td><td>3.48</td><td>2.88</td><td>2.35</td></tr><tr><td>P/S (倍)</td><td>4.59</td><td>2.65</td><td>2.02</td><td>1.61</td></tr><tr><td>净资产收益率</td><td>16.3%</td><td>19.1%</td><td>21.5%</td><td>23.2%</td></tr></table></body></html>

资料来源：东北证券股息收益率 $\mathbf{\Sigma}=\mathbf{\Sigma}$ （ 除权后年度每股现金红利总和 / 报告首页收盘价 ） $\ast100\%$ 。  

# 研究团队简介：  

[要Ta文ble强_I：nt东ro北du证cti券on通] 信行业首席分析师。格拉斯哥大学硕士，拥有 3 年军工、通信产业一级市场投资经验以及航天产业从业经验，2020 年加入东北证券，担任军工行业分析师，2023 年担任通信行业首席分析师。执业证书编号：S0550523010004。刘云坤：伦敦政治经济学院风险与金融硕士，中央财经大学金融学本科。2022 年加入东北证券，现任东北证券通信行业分析师。  

# 分析师声明  

作者具有中国证券业协会授予的证券投资咨询执业资格，并在中国证券业协会注册登记为证券分析师。本报告遵循合规、客观、专业、审慎的制作原则，所采用数据、资料的来源合法合规，文字阐述反映了作者的真实观点，报告结论未受任何第三方的授意或影响，特此声明。  

投资评级说明  


<html><body><table><tr><td rowspan="5">股票 投资 评级 说明</td><td>买入</td><td>未来6个月内，股价涨幅超越市场基准15%以上。</td><td rowspan="5">投资评级中所涉及的市场基准： A股市场以沪深300指数为市场基</td></tr><tr><td>增持</td><td>未来6个月内，股价涨幅超越市场基准5%至15%之间。</td></tr><tr><td>中性</td><td>未来6个月内，股价涨幅介于市场基准-5%至5%之间。</td></tr><tr><td>减持</td><td>未来6个月内，股价涨幅落后市场基准5%至15%之间。</td></tr><tr><td>卖出</td><td>未来6个月内，股价涨幅落后市场基准15%以上。</td></tr><tr><td rowspan="3">行业 投资 评级 说明</td><td>优于大势</td><td>未来6个月内，行业指数的收益超越市场基准。</td><td rowspan="3">场以摩根士丹利中国指数为市场基 准；美国市场以纳斯达克综合指数或 标普500指数为市场基准。</td></tr><tr><td>同步大势</td><td>未来6个月内，行业指数的收益与市场基准持平。</td></tr><tr><td>落后大势</td><td>未来6个月内，行业指数的收益落后于市场基准。</td></tr></table></body></html>  

# 重要声明  

本报告由东北证券股份有限公司（以下称“本公司”）制作并仅向本公司客户发布，本公司不会因任何机构或个人接收到本报告而视其为本公司的当然客户。  

本公司具有中国证监会核准的证券投资咨询业务资格。  

本报告中的信息均来源于公开资料，本公司对这些信息的准确性和完整性不作任何保证。报告中的内容和意见仅反映本公司于发布本报告当日的判断，不保证所包含的内容和意见不发生变化。  

本报告仅供参考，并不构成对所述证券买卖的出价或征价。在任何情况下，本报告中的信息或所表述的意见均不构成对任何人的证券买卖建议。本公司及其雇员不承诺投资者一定获利，不与投资者分享投资收益，在任何情况下，我公司及其雇员对任何人使用本报告及其内容所引发的任何直接或间接损失概不负责。  

本公司或其关联机构可能会持有本报告中涉及到的公司所发行的证券头寸并进行交易，并在法律许可的情况下不进行披露；  
可能为这些公司提供或争取提供投资银行业务、财务顾问等相关服务。  

本报告版权归本公司所有。未经本公司书面许可，任何机构和个人不得以任何形式翻版、复制、发表或引用。如征得本公司同意进行引用、刊发的，须在本公司允许的范围内使用，并注明本报告的发布人和发布日期，提示使用本报告的风险。若本公司客户（以下称“该客户”）向第三方发送本报告，则由该客户独自为此发送行为负责。提醒通过此途径获得本报告的投资者注意，本公司不对通过此种途径获得本报告所引起的任何损失承担任何责任。  

# 东北证券股份有限公司  

网址：http://www.nesc.cn 电话：95360,400-600-0686 研究所公众号：dbzqyanjiusuo  


<html><body><table><tr><td>地址</td><td>邮编</td></tr><tr><td>中国吉林省长春市生态大街6666号</td><td>130119</td></tr><tr><td>中国北京市西城区锦什坊街28号恒奥中心D座</td><td>100033</td></tr><tr><td>中国上海市浦东新区杨高南路799号</td><td>200127</td></tr><tr><td>中国深圳市福田区福中三路1006号诺德中心34D</td><td>518038</td></tr><tr><td>中国广东省广州市天河区洗村街道黄埔大道西122号之二星辉中心15楼</td><td>510630</td></tr></table></body></html>  
"""


    final_result = custom_chunk_pipeline(text2)
    print("\n===== 最终 chunks_list =====")
    for i, (is_table, ck) in enumerate(final_result):
        print(f"[{i}] is_table={is_table}, length={len(ck)}")
        print(ck)
        print('-----------------------------Split------------------------------------\n\n')

    print('--------------END-------------------')