import modal

app = modal.App("company-policy-bot")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "fastapi"
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=1200,
)
def generate(prompt: str):
    from transformers import AutoTokenizer
    from transformers import AutoModelForCausalLM

    model_id = "mdshakirkaif/company-policy-bot-merged"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )
@app.local_entrypoint()
def main():
    result = generate.remote(
        "What is the company remote work policy?"
    )

    print("\n=== MODEL OUTPUT ===\n")
    print(result)
@app.fastapi_endpoint(method="POST")
def infer(data: dict):
    result = generate.local(
        data["prompt"]
    )

    return {
        "response": result
    }