from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

text = """
Artificial intelligence is transforming modern industries.
Machine learning models help organizations make decisions.
Cloud computing provides scalable infrastructure.
Cybersecurity protects digital systems from attacks.
"""

result = summarizer(
    text,
    max_length=50,
    min_length=20,
    do_sample=False
)

print(result[0]["summary_text"])