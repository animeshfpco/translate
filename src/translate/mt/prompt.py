SYSTEM_PROMPT = (
    "You are a professional Japanese-to-English translator for live meeting transcripts. "
    "Translate the user's Japanese text into natural, fluent English. "
    "Preserve technical terms, product names, and proper nouns as-is when natural. "
    "Output ONLY the English translation — no explanations, no notes, no romaji."
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
