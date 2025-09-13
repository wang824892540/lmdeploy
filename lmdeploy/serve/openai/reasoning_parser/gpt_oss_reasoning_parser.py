# Copyright (c) OpenMMLab. All rights reserved.
# Modified from the DeepSeek-R1 parser above.
import re
from typing import Optional, Sequence, Tuple, Union

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage
from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(name='gpt-oss')
class GptOssReasoningParser(ReasoningParser):
    """Reasoning parser for gpt-oss models.

    gpt-oss uses special channel tags to separate reasoning and final answer:
        <|start|>assistant<|channel|>analysis<|message|>...<|end|>
        <|start|>assistant<|channel|>final<|message|>...<|end|>
    """

    # 预编译正则，非流式一次性提取两段
    _RE_ANALYSIS = re.compile(
        r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
        re.DOTALL,
    )
    _RE_FINAL = re.compile(
        r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|end\|>',
        re.DOTALL,
    )

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        # gpt-oss 不需要特殊 token id，因此不做 token 检查
        if not self.model_tokenizer:
            raise ValueError(
                'The model tokenizer must be passed to the ReasoningParser '
                'constructor during construction.'
            )

    # ----------------------------------------------------------
    # 流式抽取
    # ----------------------------------------------------------
    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        **kwargs,
    ) -> Union[DeltaMessage, None]:
        """
        极简策略：
        1. 把本次 delta 直接追加到“当前段”缓存。
        2. 如果 delta 里出现 <|end|>，就把缓存区 flush 成对应字段。
        3. 通道切换靠 <|channel|> 标签，出现即切换缓存目标。
        """
        # 用对象级缓存，避免跨 chunk 信息丢失
        if not hasattr(self, '_cache'):
            self._cache: str = ''
            self._in_analysis: bool = False  # 当前是否处于 analysis 通道

        self._cache += delta_text

        # 如果还没决定通道，先判断
        if not self._in_analysis and '<|channel|>analysis<|message|>' in self._cache:
            self._in_analysis = True
            # 把标签本身从缓存去掉，避免下游拿到脏数据
            self._cache = self._cache.split('<|channel|>analysis<|message|>', 1)[1]
        elif not self._in_analysis and '<|channel|>final<|message|>' in self._cache:
            self._in_analysis = False
            self._cache = self._cache.split('<|channel|>final<|message|>', 1)[1]

        # 如果碰到 <|end|> 就 flush
        if '<|end|>' in self._cache:
            seg, self._cache = self._cache.split('<|end|>', 1)
            if self._in_analysis:
                return DeltaMessage(reasoning_content=seg)
            else:
                return DeltaMessage(content=seg)

        # 还没结束，继续累积
        if self._in_analysis:
            return DeltaMessage(reasoning_content=delta_text)
        return DeltaMessage(content=delta_text)

    # ----------------------------------------------------------
    # 非流式抽取
    # ----------------------------------------------------------
    def extract_reasoning_content(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        一次性返回 (reasoning_content, final_content)
        如果某一段缺失则对应 None
        """
        reasoning_match = self._RE_ANALYSIS.search(model_output)
        final_match = self._RE_FINAL.search(model_output)

        reasoning_content = reasoning_match.group(1) if reasoning_match else None
        final_content = final_match.group(1) if final_match else None
        return reasoning_content, final_content
