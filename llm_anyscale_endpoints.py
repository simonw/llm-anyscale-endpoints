import click
import httpx
import json
import llm
from llm.default_plugins.openai_models import Chat

MODELS = (
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-3-8b-chat-hf",
    "meta-llama/Llama-3-70b-chat-hf",
    "codellama/CodeLlama-70b-Instruct-hf",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mlabonne/NeuralHermes-2.5-Mistral-7B",
    "google/gemma-7b-it"
)


class AnyscaleEndpointChat(Chat):
    needs_key = "anyscale-endpoints"

    def __str__(self):
        return "AnyscaleEndpoints: {}".format(self.model_id)


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="anyscale-endpoints")
    def anyscale_endpoints():
        "llm-anyscale-endpoints plugin commands"

    @anyscale_endpoints.command(name="refresh")
    def refresh():
        "Refresh the list of models from the Anyscale Endpoints API"
        key = llm.get_key("", "anyscale-endpoints", "LLM_ANYSCALE_ENDPOINTS_KEY")
        if not key:
            raise click.ClickException("No key found for Anyscale Endpoints")
        headers = {"Authorization": f"Bearer {key}"}
        response = httpx.get(
            "https://api.endpoints.anyscale.com/v1/models", headers=headers
        )
        response.raise_for_status()
        models = response.json()["data"]
        # Filter out just the ones with "rayllm_metadata": {"engine_config": {"model_type": "text-generation"}}
        text_generation = [
            model["id"]
            for model in models
            if model.get("rayllm_metadata", {})
            .get("engine_config", {})
            .get("model_type")
            == "text-generation"
        ]
        if not text_generation:
            raise click.ClickException("No text-generation models found")
        path = llm.user_dir() / "llm-anyscale-endpoints.json"
        path.write_text(json.dumps(text_generation, indent=4))
        click.echo("{} models saved to {}".format(len(text_generation), path), err=True)
        click.echo(json.dumps(text_generation, indent=4))


@llm.hookimpl
def register_models(register):
    # Only do this if the key is set
    key = llm.get_key("", "anyscale-endpoints", "LLM_ANYSCALE_ENDPOINTS_KEY")
    if not key:
        return
    # If llm-anyscale-endpoints.json exists, use that - otherwise use MODELS
    path = llm.user_dir() / "llm-anyscale-endpoints.json"
    if path.exists():
        model_ids = json.loads(path.read_text())
    else:
        model_ids = MODELS

    for model_id in model_ids:
        register(
            AnyscaleEndpointChat(
                model_id=model_id,
                model_name=model_id,
                api_base="https://api.endpoints.anyscale.com/v1",
            )
        )
