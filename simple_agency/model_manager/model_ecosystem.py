from typing import Any, Dict, List, Union
import tiktoken


def num_tokens_from_messages(messages, model="gpt-4"):
    """ Returns the number of tokens used by a list of messages.

    Args:
        messages (List[Dict[str, str]]):
            - A list of messages, each of which is a dictionary with keys "role", "name", and "content".
                - The "role" key is optional and defaults to "user".
                - The "name" key is optional and defaults to "user" if "role" is "user" and "assistant" otherwise.
                - The "content" key is required and must be a string.
        model (str, optional): The model to use. Defaults to "gpt-4".

    Returns:
        int: The number of tokens used by the messages.

    Example Usage:

    ```
    model = "gpt-3.5-turbo-0613"
    messages = [
        {"role": "system", "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."},
        {"role": "system", "name":"example_user", "content": "New synergies will help drive top-line growth."},
        {"role": "system", "name": "example_assistant", "content": "Things working well together will increase revenue."},
        {"role": "system", "name":"example_user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
        {"role": "system", "name": "example_assistant", "content": "Let's talk later when we're less busy about how to do better."},
        {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
    ]
    print(f"{num_tokens_from_messages(messages, model)} prompt tokens counted.") # Should show ~126 total_tokens
    ```

    Raises:
        NotImplementedError: If the model itself is not supported.
        KeyError: If the encoding for a particular model is not supported/found
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # note: future models may deviate from
    if any(model.startswith(x) for x in ["gpt-3.5", "gpt-4"]):
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to 
  tokens.""")
