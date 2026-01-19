import re

def clean_text(t: str) -> str:
    t = (t or "").strip()
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = t.replace("#", " ")
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_input(entity: str, text: str) -> str:
    # Entity-aware input format
    return f"entity: {clean_text(entity)} [SEP] tweet: {clean_text(text)}"
