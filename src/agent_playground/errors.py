class AgentPlaygroundError(Exception):
    """项目内所有自定义异常的基类，方便统一捕获。"""


class ConfigError(AgentPlaygroundError):
    """配置缺失或非法。"""


class LLMCallError(AgentPlaygroundError):
    """调用大模型接口失败，可携带状态码供上层决定是否重试。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
