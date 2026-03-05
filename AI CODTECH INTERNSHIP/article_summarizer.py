# Article Summarizer using NLP
# Prajwal - NLP Project

from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example lengthy article
article = """
Artificial Intelligence is transforming industries across the world.
From healthcare to finance, AI systems are improving efficiency and
helping humans solve complex problems. Machine learning models can
analyze large datasets quickly and identify patterns that humans might
miss.

In healthcare, AI helps doctors diagnose diseases earlier by analyzing
medical images and patient data. In finance, algorithms detect fraud
and predict market trends. Self-driving cars rely heavily on AI to
interpret sensor data and navigate roads safely.

Despite its benefits, AI also raises concerns about job displacement,
privacy issues, and ethical decision making. Experts believe the key
challenge is ensuring AI systems remain transparent, fair, and aligned
with human values.

As technology advances, collaboration between policymakers,
researchers, and businesses will be essential to ensure AI benefits
society while minimizing risks.
"""

# Generate summary
summary = summarizer(
    article,
    max_length=120,
    min_length=40,
    do_sample=False
)

# Display results
print("\n================ ORIGINAL ARTICLE ================\n")
print(article)

print("\n================ SUMMARY ================\n")
print(summary[0]['summary_text'])