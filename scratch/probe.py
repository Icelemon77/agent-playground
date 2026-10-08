"""一次性探测脚本：验证 key / base_url / 模型名。跑完可删。"""

import json

import httpx

from agent_playground.config import Settings


def main() -> int:
    s = Settings()
    url = f"{s.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": s.llm_model,
        "messages": [
            {"role": "system", "content": "你是一个简洁的助手，回答不超过 20 个字。"},
            {"role": "user", "content": "用一句话介绍你自己"},
        ],
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {s.llm_api_key}",
        "Content-Type": "application/json",
    }
    payload["reasoning_effort"] = (
        "none"  # 取消思考模式的调用，若要用思考模式，删掉这句话即可
    )
    with httpx.Client(timeout=s.request_timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        print("HTTP 状态码:", resp.status_code)
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if resp.status_code == 200:
            print("\n模型回答:", data["choices"][0]["message"]["content"])
            print("token 用量:", data.get("usage"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
