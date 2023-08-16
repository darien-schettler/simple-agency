from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import BaseCallbackHandler, StreamingStdOutCallbackHandler
from boiler_llm_app.backend.config.models import (
    _OPEN_AI_MODEL_NAMES,
    _META_MODEL_NAMES,
    _ANTHROPIC_MODEL_NAMES,
    _HUGGINGFACE_MODEL_NAMES,
    get_hf_model_name
)


def get_openai_chat_model(model_name="gpt-3.5-turbo-0613",
                          temperature=0.7,
                          use_streaming=False,
                          verbose=True,
                          **kwargs):
    """ TBD """
    chat_model = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        verbose=verbose,
        streaming=use_streaming,
        callbacks=[StreamingStdOutCallbackHandler()] if use_streaming else None,
        **kwargs
    )

    return chat_model


# ssh paperspace@172.83.15.60
def get_meta_chat_model(model_name="llama2-70b", remote_ip="172.83.15.60", **kwargs):
    """ TBD """


def get_anthropic_chat_model(model_name, param):
    """ TBD """
    raise NotImplementedError("... Anthropic models not implemented yet ...")

def get_huggingface_chat_model(model_name, is_chat=True, **kwargs):
    """ TBD """
    hf_model_str = get_hf_model_name(model_name, is_chat=True)


    raise NotImplementedError("... Huggingface models not implemented yet ...")

def chat_model_from_name(model_name="gpt-4", **kwargs):
    """ Returns an instantiated model wrapped as a chat model (Langchain) source based on name

    Possible options and their respective providers include:
        - "gpt-4"            - 'open_ai'
        - "gpt-3.5-turbo"    - 'open_ai'
        - "gpt-3.5-turbo-16" - 'open_ai'
        - "llama2-70b"       - 'meta'
        - "llama2-7b"        - 'meta'
        - "llama2-3b"        - 'meta'
        - "claude2"          - 'anthropic'
        - "hf-gpt2"          - 'huggingface'
        - "hf-*"             - 'huggingface' (any model name starting with "hf-")

    Note that all open_ai and anthropic options will require a valid API key as access will be via API.
    For meta and huggingface options, the model will be downloaded and hosted on a private external instance
    this will take some time the first time it happens, but the model will subsequently be cached.

    Args:
        model_name (str): The name of the model to load
        **kwargs: Additional keyword arguments to pass to the model loader

    Returns:
        langchain.chat_models.ChatModel: The instantiated chat model
    """

    if model_name.lower() in _OPEN_AI_MODEL_NAMES:
        return get_openai_chat_model(model_name=model_name, **kwargs)
    elif model_name.lower() in _META_MODEL_NAMES:
        # we have access to the llama models via huggingface
        if model_name.lower().startswith("ll"):
            return get_huggingface_chat_model(model_name=model_name, **kwargs)
        else:
            return get_meta_chat_model(model_name=model_name, **kwargs)

    elif model_name.lower() in _ANTHROPIC_MODEL_NAMES:
        return get_anthropic_chat_model(model_name=model_name, **kwargs)
    elif model_name.lower() in _HUGGINGFACE_MODEL_NAMES:
        return get_huggingface_chat_model(model_name=model_name, **kwargs)
    else:
        raise NotImplementedError(f"\n... Model name {model_name} not implemented yet ...\n")
