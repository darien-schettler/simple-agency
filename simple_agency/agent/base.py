from boiler_llm_app.backend.prompting.library import prompt_library
from boiler_llm_app.backend.tools import (
    llm_tools,
    #basic_tools,
    #api_tools,
    #utility_tools,
    #test_tools
)


class BaseAgent:
    def __init__(self, model_name, stop_string="Observation: ", agent_name=None, version="v1"):
        self.model_name = model_name
        self.agent_name = f'agent_{version}' if not agent_name else agent_name
        self.stop_string = stop_string
        self.prompt_template = prompt_library.get_prompt_str(f'agent_{version}')
        self.prompt_vars = prompt_library.get_prompt_vars(f'agent_{version}')

        # Initialize
        self.chat_history = []
        self.tools = []
        self.variables = []
        self.scratchpad = ""


    def send(self, *args, **kwargs):
        pass




