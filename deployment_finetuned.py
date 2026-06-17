# import modal
# from fastapi import FastAPI
# from pydantic import BaseModel

# app = modal.App("company-policy-bot")

# image = (
#     modal.Image.debian_slim()
#     .pip_install(
#         "torch",
#         "transformers",
#         "accelerate",
#         "huggingface_hub",
#         "fastapi",
#         "pydantic",
#     )
# )

# MODEL_ID = "mdshakirkaif/company-policy-bot-merged"


# @app.cls(
#     image=image,
#     gpu="T4",
#     timeout=1200,
#     scaledown_window=1800,  # Keep warm for 30 min
# )
# class ModelServer:

#     @modal.enter()
#     def load_model(self):
#         from transformers import AutoTokenizer, AutoModelForCausalLM

#         print("Loading tokenizer...")

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             MODEL_ID
#         )

#         print("Loading model...")

#         self.model = AutoModelForCausalLM.from_pretrained(
#             MODEL_ID,
#             device_map="auto"
#         )

#         print("Model loaded successfully!")

#     @modal.method()
#     def generate(self, prompt: str):

#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt"
#         ).to(self.model.device)

#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=100,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             pad_token_id=self.tokenizer.eos_token_id,
#         )

#         response = self.tokenizer.decode(
#             outputs[0],
#             skip_special_tokens=True
#         )

#         return response


# model_server = ModelServer()

# web_app = FastAPI()


# class PromptRequest(BaseModel):
#     prompt: str


# @web_app.post("/")
# def infer(request: PromptRequest):

#     response = model_server.generate.remote(
#         request.prompt
#     )

#     return {
#         "response": response
#     }


# @app.function()
# @modal.asgi_app()
# def api():
#     return web_app


## deployed but for every request it use to 1. Starts a container
#2. Loads a 6+ GB model
#3. Generates text
#4. Returns response

# This is very slow and expensive. so we are using above code to 
import modal
from fastapi import FastAPI

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
    from transformers import AutoTokenizer, AutoModelForCausalLM

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
    max_new_tokens=150,
    do_sample=False,
    temperature=None,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


web_app = FastAPI()

@web_app.post("/")
def infer(data: dict):
    result = generate.remote(data["prompt"])
    return {"response": result}


@app.function(image=image)
@modal.asgi_app()
def api():
    return web_app