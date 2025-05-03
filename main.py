import typer
import requests
import os

app = typer.Typer()

MODEL_MAP = {
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "gpt2": "gpt2"
}

def query_huggingface(prompt, model, max_tokens, temperature, hf_token):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": f"<s>[INST] {prompt} [/INST]",
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"API Error {response.status_code}: {response.text}")
    return response.json()[0]["generated_text"]

@app.command()
def ask(
    prompt: str = typer.Option(..., prompt="Prompt"),
    max_tokens: int = typer.Option(100),
    temperature: float = typer.Option(0.9)
):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        typer.echo("Missing HF_TOKEN environment variable.")
        raise typer.Exit()

    model_name = typer.prompt("Model (zephyr / gpt2)").lower()
    model = MODEL_MAP.get(model_name)
    if not model:
        typer.echo("Invalid model.")
        raise typer.Exit()
    typer.echo(f"Sending request to Hugging Face API using model: {model_name}...")
    try:
        response = query_huggingface(prompt, model, max_tokens, temperature, hf_token)
        typer.echo(f"AI: {response.strip()}")
    except Exception as e:
        typer.echo(f"Error: {e}")

if __name__ == "__main__":
    app()
