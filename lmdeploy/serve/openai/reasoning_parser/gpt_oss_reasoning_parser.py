# Copyright (c) OpenMMLab. All rights reserved.
# 极简“零剔除”版 gpt-oss reasoning parser
import re
from typing import Optional, Sequence, Tuple, Union

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage
from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(name='gpt-oss')
class GptOssReasoningParser(ReasoningParser):
    """零剔除：标签原样保留，只按通道分流."""

    # 非流式正则，同样不做任何剔除
    _RE_ANALYSIS = re.compile(
        r'(<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>)',
        re.DOTALL,
    )
    _RE_FINAL = re.compile(
        r'(<\|start\|>assistant<\|channel\|>final<\|message\|>.*?<\|end\|>)',
        re.DOTALL,
    )

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        if not self.model_tokenizer:
            raise ValueError('Tokenizer must be provided.')

    # ---------- 流式 ----------
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
        # 用对象级缓存
        if not hasattr(self, '_buf'):
            self._buf: str = ''
            self._in_analysis: Optional[bool] = None  # None=未确定, True/False=已确定

        self._buf += delta_text

        # 通道判定（只做判定，不剔除）
        if self._in_analysis is None:
            if '<|channel|>analysis<|message|>' in self._buf:
                self._in_analysis = True
            elif '<|channel|>final<|message|>' in self._buf:
                self._in_analysis = False

        # 碰到 <|end|> 就整段（含标签）flush
        if '<|end|>' in self._buf:
            seg, self._buf = self._buf.split('<|end|>', 1)
            seg += '<|end|>'          # 把分隔符加回来
            if self._in_analysis:
                return DeltaMessage(reasoning_content=seg)
            else:
                return DeltaMessage(content=seg)

        # 还没结束，按当前通道继续累积
        if self._in_analysis is True:
            return DeltaMessage(reasoning_content=delta_text)
        elif self._in_analysis is False:
            return DeltaMessage(content=delta_text)
        else:
            # 通道都还没出现，先全部当成 reasoning（也可以返回 None 继续攒）
            return DeltaMessage(reasoning_content=delta_text)

    # ---------- 非流式 ----------
    def extract_reasoning_content(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> Tuple[Optional[str], Optional[str]]:
        reasoning_match = self._RE_ANALYSIS.search(model_output)
        final_match = self._RE_FINAL.search(model_output)

        reasoning_content = reasoning_match.group(0) if reasoning_match else None
        final_content = final_match.group(0) if final_match else None
        return reasoning_content, final_content
