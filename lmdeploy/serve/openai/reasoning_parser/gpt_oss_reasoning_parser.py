# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Optional, Sequence, Tuple, Union

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage
from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(name='gpt-oss')
class GptOssReasoningParser(ReasoningParser):
    """Reasoning parser for gpt-oss template."""

    # 模板里我们自己约定的通道标签
    _ANALYSIS_START = "<|start|>assistant<|channel|>analysis<|message|>"
    _ANALYSIS_END   = "<|end|>"
    _FINAL_START    = "<|start|>assistant<|channel|>final<|message|>"
    _FINAL_END      = "<|end|>"

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        # 预编译正则：一次性抽取 analysis / final 内容
        self._full_regex = re.compile(
            rf"{re.escape(self._ANALYSIS_START)}(.*?){re.escape(self._ANALYSIS_END)}"
            rf".*?"
            rf"{re.escape(self._FINAL_START)}(.*?){re.escape(self._FINAL_END)}",
            re.DOTALL,
        )
        # 流式状态机
        self._inside_analysis = False

    # ------------------------------------------------------------------
    # 非流式：整段返回
    # ------------------------------------------------------------------
    def extract_reasoning_content(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> Tuple[Optional[str], Optional[str]]:
        m = self._full_regex.search(model_output)
        if not m:
            # 退化：没有 analysis，整段当 final
            return None, model_output
        reasoning_content, final_content = m.groups()
        return reasoning_content or None, final_content or None

    # ------------------------------------------------------------------
    # 流式：逐 delta 解析
    # ------------------------------------------------------------------
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
        # 简单实现：按字符串级状态机即可，token_id 版可同理扩展
        if not delta_text:
            return None

        # 状态机：是否已进入到 analysis 段
        if not self._inside_analysis:
            if self._ANALYSIS_START in delta_text:
                self._inside_analysis = True
                # 把标签之后的内容作为 reasoning 起点
                _, _, after = delta_text.partition(self._ANALYSIS_START)
                if after:
                    return DeltaMessage(reasoning_content=after)
                return None
            # 还没进 analysis，直接丢弃
            return None

        # 当前在 analysis 段
        if self._ANALYSIS_END in delta_text:
            self._inside_analysis = False
            # 标签之前的是 reasoning，之后的是 final
            reasoning, _, remain = delta_text.partition(self._ANALYSIS_END)
            final = remain.partition(self._FINAL_START)[2]  # 去掉 final 头
            return DeltaMessage(
                reasoning_content=reasoning or None,
                content=final or None,
            )

        # 仍在 analysis，全部给 reasoning
        return DeltaMessage(reasoning_content=delta_text)

    # ------------------------------------------------------------------
    # 多轮对话时需要 reset 状态
    # ------------------------------------------------------------------
    def reset(self):
        self._inside_analysis = False
