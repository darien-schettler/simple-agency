from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import BaseCallbackHandler, StreamingStdOutCallbackHandler
from simple_agency.config.models import (
    OPEN_AI_MODEL_NAMES,
    META_MODEL_NAMES,
    ANTHROPIC_MODEL_NAMES,
    HUGGINGFACE_MODEL_NAMES,
    get_hf_model_name
)
from simple_agency.runhouse_ops.instance_handler import HFChatModel
import runhouse as rh


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


def get_huggingface_chat_model(model_name, is_chat=True, rh_loader=True, force_rh_restart=False,
                               model_overrides=None, **kwargs):
    """ TBD """
    hf_model_str = get_hf_model_name(model_name, is_chat=True)

    # TODO: Config file
    _default_chat_model_config = dict(
        device = "cuda",
        trust_remote_code = True,
        model_inference_temperature = 0.7,
        model_inference_max_tokens = 4000,
        model_inference_repetition_penalty = 1.0,
        model_inference_returns_full_text = True,
        pipe_task = "text-generation",
        streaming = True
    )
    model_overrides = {"model_id": hf_model_str, **model_overrides} if model_overrides is not None else {"model_id": hf_model_str}
    chat_model_config = {**_default_chat_model_config, **model_overrides}

    # TODO: VLLM (IMPROVE LATER FOR PERFORMANCE)
    if rh_loader:

        gpu = rh.cluster("a100-cluster")

        if force_rh_restart:
            gpu.restart_server(restart_ray=True, resync_rh=False)

        if not f"hf-{model_name}-model" in gpu.list_keys():
            gpu.delete_keys("all")
            chat_model = HFChatModel(**chat_model_config).get_or_to("a100-cluster", name=f"hf-{model_name}-model")
        else:
            chat_model = gpu.get(f"hf-{model_name}-model", remote=True)
    else:
        raise NotImplementedError("... Local huggingface model loading not implemented yet ...")

    return chat_model


def chat_model_from_name(model_name="gpt-4", **kwargs):
    """ Returns an instantiated model wrapped as a chat model (Langchain) source based on name

    Possible options and their respective providers include:
        - "gpt-4"            - 'open_ai'
        - "gpt-3.5-turbo"    - 'open_ai'
        - "gpt-3.5-turbo-16" - 'open_ai'
        - "llama2-70b"       - 'meta'
        - "llama2-13b"       - 'meta'
        - "llama2-7b"        - 'meta'
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

    if model_name.lower() in OPEN_AI_MODEL_NAMES:
        return get_openai_chat_model(model_name=model_name, **kwargs)
    elif model_name.lower() in META_MODEL_NAMES:
        # we have access to the llama models via huggingface
        if model_name.lower().startswith("ll"):
            return get_huggingface_chat_model(model_name=model_name, **kwargs)
        else:
            return get_meta_chat_model(model_name=model_name, **kwargs)

    elif model_name.lower() in ANTHROPIC_MODEL_NAMES:
        return get_anthropic_chat_model(model_name=model_name, **kwargs)
    elif model_name.lower() in HUGGINGFACE_MODEL_NAMES:
        return get_huggingface_chat_model(model_name=model_name, **kwargs)
    else:
        raise NotImplementedError(f"\n... Model name {model_name} not implemented yet ...\n")
