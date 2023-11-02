import llm
from llm.default_plugins.openai_models import Chat

MODELS = (
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "mistralai/Mistral-7B-Instruct-v0.1"
)


class AnyscaleEndpointChat(Chat):
    needs_key = "anyscale-endpoints"

    def __str__(self):
        return "AnyscaleEndpoints: {}".format(self.model_id)


@llm.hookimpl
def register_models(register):
    # Only do this if the key is set
    key = llm.get_key("", "anyscale-endpoints", "LLM_ANYSCALE_ENDPOINTS_KEY")
    if not key:
        return
    for model_id in MODELS:
        register(
            AnyscaleEndpointChat(
                model_id=model_id,
                model_name=model_id,
                api_base="https://api.endpoints.anyscale.com/v1",
            )
        )
