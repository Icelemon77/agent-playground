from agent_playground.model_router import CHEAP, STRONG, pick_model


def test_pick_model_routes_simple_question_to_cheap() -> None:
    assert pick_model("把 hello 翻译成中文") == CHEAP

def test_pick_model_routes_complex_question_to_strong() -> None:
    assert pick_model("对比两种 RAG 切块方案的优劣并给出取舍") == STRONG