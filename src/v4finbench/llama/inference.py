import math


def extract_yes_no(text: str) -> str | None:
    text_upper = text.upper()
    yes_idx = text_upper.find("YES")
    no_idx = text_upper.find("NO")
    if yes_idx >= 0 and no_idx >= 0:
        return "YES" if yes_idx < no_idx else "NO"
    if yes_idx >= 0:
        return "YES"
    if no_idx >= 0:
        return "NO"
    return None


def normalize_yes_no_logprobs(logp_yes: float, logp_no: float) -> tuple[float, float]:
    max_logp = max(logp_yes, logp_no)
    yes_exp = math.exp(logp_yes - max_logp)
    no_exp = math.exp(logp_no - max_logp)
    denom = yes_exp + no_exp
    return yes_exp / denom, no_exp / denom
