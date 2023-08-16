def split_raw_llm_response(raw_response, top_k_sources, return_llm_response=True):
    """ Parses the model's response and returns source information (and optionally the model's response)

    Args:
        raw_response (str): A dictionary containing the raw LLM response and the source Documents.
        top_k_sources (List[Document]): A list of top_k Documents that have been chunked w/ appropriate metadata.
        return_llm_response (bool, optional): Whether to return only sources or include the LLM response as well

    Returns:
        List[Document]: A list of Documents that are the sources for the answer.
    """
    llm_response, sources = raw_response, ""
    if "SOURCES: " in raw_response:
        response_split = raw_response.split("SOURCES: ")
        llm_response, sources = response_split[0], "".join(response_split[1:])

    source_keys = [x.strip() for x in sources.split(",")]
    referenced_sources = [doc for doc in top_k_sources if doc.metadata["source"] in source_keys]

    if return_llm_response:
        return llm_response, referenced_sources
    else:
        return referenced_sources
