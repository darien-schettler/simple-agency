""" Placeholder for various pieces of model configuration that we want to use elsewhere """
OPEN_AI_MODEL_NAMES = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
META_MODEL_NAMES = ["llama2-70b", "llama2-13b", "llama2-7b"]
ANTHROPIC_MODEL_NAMES = ["claude2"]
HUGGINGFACE_MODEL_NAMES = ["hf-gpt2"]

_HF_MODEL_NAME_MAP = {
    "chat":{
        "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
        "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
        "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
        "qwen-7b":"Qwen/Qwen-7B-Chat"
    },
    "base":{
        "llama2-70b": "meta-llama/Llama-2-70b-hf",
        "llama2-13b": "meta-llama/Llama-2-13b-hf",
        "llama2-7b": "meta-llama/Llama-2-7b-hf",
        "qwen-7b":"Qwen/Qwen-7B"
    },
}


def get_hf_model_name(model_name, is_chat=True):
    """ Returns the model name as it should be specified in the huggingface model repo """
    hf_model_str = _HF_MODEL_NAME_MAP["chat" if is_chat else "base"].get(model_name)
    if hf_model_str is None:
        raise ValueError(f"Model name {model_name} not found in the huggingface model map."
                         f"The available {'chat' if is_chat else 'base'} models are:\n"
                         f"{sorted(_HF_MODEL_NAME_MAP['chat' if is_chat else 'base'].keys())}")
    return hf_model_str

