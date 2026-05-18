SYSTEM_PROMPT = (
    "You are a professional Japanese-to-English translator for live meeting transcripts. "
    "Translate ONLY the current user message into natural, fluent English. "
    "Prior conversation turns are context only — do NOT repeat or continue them. "
    "Preserve technical terms, product names, and proper nouns as-is when natural. "
    "Output ONLY the English translation of the current message — no explanations, no notes, no romaji."
)


def build_messages(
    text_ja: str,
    context: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build chat-template messages for an MT call.

    `context` is a rolling window of recent (ja, en) pairs replayed as prior
    turns so the model carries terminology/pronouns across utterances.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ja, en in context or []:
        messages.append({"role": "user", "content": ja})
        messages.append({"role": "assistant", "content": en})
    messages.append({"role": "user", "content": text_ja})
    return messages
