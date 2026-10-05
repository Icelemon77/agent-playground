from collections.abc import Iterable, Mapping


def count_tokens_approx(text:str|None, chars_per_token:int=4)->int:
    if not text:
        return 0
    return max(1, len(text) // chars_per_token)


def truncate(text:str, max_chars:int=100, suffix:str="...")->str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def merge_config(base: Mapping[str, object],override: Mapping[str, object] | None = None,) -> dict[str, object]:
    result:dict[str,object] = dict(base)
    if override:
        result.update(override)
    return result


def parse_env_pairs(lines:Iterable[str])->dict[str,str]:
    pairs:dict[str,str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        pairs[key.strip()] = value.strip()
    return pairs


def build_llm_headers(api_key:str|None=None, extra:Mapping[str,str]|None=None):
    headers:dict[str,str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers