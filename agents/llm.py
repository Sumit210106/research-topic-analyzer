from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)


def summarize(text: str) -> str:
    """
    Takes a long text and returns a summary.
    """

    try:

        text = text[:1000]

        summary = summarizer(
            text,
            max_length=120,
            min_length=30,
            do_sample=False
        )

        return summary[0]["summary_text"]

    except Exception as e:
        print("❌ LLM Error:", e)


        return text[:200]


def summarize_all(texts: list) -> list:
    """
    Takes list of texts and returns list of summaries.
    """

    summaries = []

    for t in texts:
        summaries.append(summarize(t))

    return summaries