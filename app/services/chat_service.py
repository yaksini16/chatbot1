"""
FAQ bot for the Legal Metrology Compliance chatbot.
Provides instant answers on Legal Metrology (Packaged Commodities) Rules, 2011 and FSSAI.
"""
from typing import Tuple, List
from pipeline.rules.engine import get_rules


def answer_compliance_question(message: str) -> Tuple[str, List[str]]:
    msg = message.lower()
    rules = get_rules().get("fields", {})
    for field_key, data in rules.items():
        keywords = [field_key.replace("_", " ")] + data.get("label", "").lower().split()
        if any(kw in msg for kw in keywords if len(kw) > 3):
            return (
                f"**{data['label']}**\n\n"
                f"{data['rule_explanation']}\n\n"
                f"• Pass: {data['pass_condition']}\n"
                f"• Fail: {data['fail_condition']}\n\n"
                f"Penalty: {data['penalty']}",
                [data['rule_citation']]
            )
    return (
        "I can answer queries about Legal Metrology (Packaged Commodities) Rules, 2011 and FSSAI. "
        "Try asking about: MRP, Net Quantity, Manufacturer Address, Expiry Date, "
        "FSSAI License, Country of Origin, or Consumer Care details.",
        ["Legal Metrology (Packaged Commodities) Rules, 2011"]
    )
