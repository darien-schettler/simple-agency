from langchain.agents import Tool
from langchain.agents import load_tools
from simple_agency.tools.configs import other_tools

def get_lc_tools(custom_descriptions=None, tools_to_use=None):
    """ Get a list of tools to use with the agent

    Args:
        custom_descriptions (list, optional): Descriptions (one per tool)
        tools_to_use (list, optional): Tool names to use (found in LC)

    """
    if tools_to_use is None:
        tools_to_use = ['python_repl', 'terminal', 'wikipedia', 'human', 'arxiv']

    if custom_descriptions is not None:
        desc_str = ""
        raise NotImplementedError("Custom descriptions not yet implemented")
    else:
        desc_str = " Keyword arguments are: 'query' representing the text we want to use with the search engine."
    return [
        Tool(
            func=x.run,
            name=x.name.title(),
            description=x.description + desc_str,
            return_direct=False,
        ) for x in load_tools(tools_to_use)
    ]


def get_other_lc_tools(tools_to_use=None):
    """ Get a list of tools to use with the agent

    Args:
        tools_to_use (list, optional): Tool names to use (found in LC)
    """
    tools_to_use = other_tools if tools_to_use is None else tools_to_use
    return ([
        Tool(**tool_kwargs) if type(tool_kwargs)==dict else tool_kwargs
        for tool_name, tool_kwargs in tools_to_use.items()
    ])


def get_tools(tools_to_use=None):
    main_lc_tools = get_lc_tools(tools_to_use=tools_to_use)
    other_lc_tools = get_other_lc_tools(tools_to_use=tools_to_use)
    return main_lc_tools + other_lc_tools

