from transformers import pipeline

# Load GPT-2 text generation model
generator = pipeline("text-generation", model="gpt2")

# Take user input
prompt = input("Enter a topic: ")

# Generate text
result = generator(
    prompt,
    max_length=120,
    num_return_sequences=1,
    temperature=0.7
)

print("\n===== GENERATED PARAGRAPH =====\n")
print(result[0]["generated_text"])