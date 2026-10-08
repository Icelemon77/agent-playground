from agent_playground.config import Settings
from agent_playground.errors import ConfigError
from agent_playground.logging_config import setup_logging


def main() -> int:
    settings = Settings()
    logger = setup_logging(settings.log_level)
    logger.info("启动 %s (env=%s)", settings.app_name, settings.env)

    try:
        if not settings.llm_api_key:
            raise ConfigError("缺少 LLM_API_KEY，请复制 .env.example 为 .env 并填写")
        logger.info("配置校验通过，请求超时=%.1fs", settings.request_timeout)
    except ConfigError as exc:
        logger.error("启动失败：%s", exc)
        return 1

    logger.info("Day 1 骨架运行成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
