from src.medvision.data import build_user_prompt


def test_prompt_includes_context():
    prompt = build_user_prompt("shortness of breath")
    assert "shortness of breath" in prompt
    assert "treatment advice" in prompt
