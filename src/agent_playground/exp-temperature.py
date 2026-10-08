"""温度实验：同一 prompt，temperature=0 与 1.5 各跑 3 次。一次性脚本。"""

import time

import httpx

from agent_playground.config import Settings

PROMPT = "用大约 100 个字写一个关于猫在雨天的小场景。"
TEMPERATURES = [1.5]
RUNS = 3


def call_once(
    client: httpx.Client, settings: Settings, temperature: float
) -> tuple[str, int, float]:
    """发一次请求，返回 (回答, 输出 token 数, 耗时秒数)。"""
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": PROMPT},
        ],
        "stream": False,
        "reasoning_effort": "none",  # 关键：非思考模式，temperature 才会生效
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    resp = client.post(url, json=payload, headers=headers)
    elapsed = time.perf_counter() - start

    data = resp.json()
    if "reasoning_content" in data["choices"][0]["message"]:
        print("  ⚠ 响应里还有 reasoning_content，说明思考模式没关掉！")

    text = data["choices"][0]["message"]["content"].strip()
    tokens = data["usage"]["completion_tokens"]
    return text, tokens, elapsed


def main() -> int:
    settings = Settings()
    with httpx.Client(timeout=settings.request_timeout) as client:
        for temp in TEMPERATURES:
            print(f"\n===== temperature = {temp} =====")
            for run in range(1, RUNS + 1):
                text, tokens, elapsed = call_once(client, settings, temp)
                print(f"[第 {run} 次] {elapsed:.1f}s / 输出 {tokens} tokens")
                print(f"  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
