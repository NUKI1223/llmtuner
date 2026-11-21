"""A rule-based stand-in for a model: reads a support ticket on stdin, prints a category.

Two versions with different rules, so the example can compare a "before" and
an "after" without calling a real model.
"""

import sys

WARRANTY = ("broke", "cracked", "snapped", "defect", "warranty")
RETURNS_V1 = ("return", "refund")
RETURNS_V2 = RETURNS_V1 + ("send back",)
SHIPPING = ("deliver", "shipping", "tracking", "arrive")
BILLING = ("charged", "invoice")


def has(text: str, words) -> bool:
    return any(w in text for w in words)


def triage_v1(text: str) -> str:
    if has(text, RETURNS_V1):
        return "returns"
    if has(text, SHIPPING):
        return "shipping"
    if has(text, WARRANTY):
        return "warranty"
    return "billing"


def triage_v2(text: str) -> str:
    # v2 checks for damage first and knows more phrases, but that new order
    # sends "arrived broken, want to return it" to warranty.
    if has(text, WARRANTY):
        return "warranty"
    if has(text, RETURNS_V2):
        return "returns"
    if has(text, SHIPPING):
        return "shipping"
    if has(text, BILLING):
        return "billing"
    return "other"


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "v1"
    ticket = sys.stdin.read().lower()
    print(triage_v2(ticket) if version == "v2" else triage_v1(ticket))
