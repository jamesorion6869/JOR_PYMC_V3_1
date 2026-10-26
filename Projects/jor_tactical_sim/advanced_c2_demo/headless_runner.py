"""
Scripted decision policies for headless (non-interactive) simulation runs.
"""


def make_input_fn(policy="intervene"):
    """
    Return an input() replacement that answers decision prompts without a human.

    Policies:
      intervene  — always choose option 1 (act / stabilize / rotate)
      ignore     — always choose option 3 (accept risk / hold course)
      alternate  — cycle 1 → 2 → 3 → 1 …
    """
    policy = (policy or "intervene").lower().strip()
    state = {"i": 0}

    def decide(prompt=""):
        text = (prompt or "").lower()
        # Scenario picker in main() — leave alone if somehow reached
        if text.startswith("scenario"):
            return "5"
        if policy == "ignore":
            return "3"
        if policy == "alternate":
            choices = ("1", "2", "3")
            sel = choices[state["i"] % 3]
            state["i"] += 1
            return sel
        # default: intervene
        return "1"

    return decide
