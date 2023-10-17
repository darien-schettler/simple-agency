
# Built-in Imports
import re, os, time, json, asyncio
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from langchain.memory import ConversationBufferWindowMemory
from langchain import SerpAPIWrapper, LLMChain
from langchain.callbacks import StdOutCallbackHandler, StreamingStdOutCallbackHandler
from langchain.agents.structured_chat.output_parser import StructuredChatOutputParserWithRetries
from simple_agency.prompting.library import prompt_library

# Langchain Imports
from langchain.agents import BaseSingleActionAgent, BaseMultiActionAgent, AgentExecutor
from langchain.agents.agent import ExceptionTool
from langchain.agents.tools import InvalidTool
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from langchain.tools.base import BaseTool
from langchain.utils.input import get_color_mapping
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate

from langchain import SerpAPIWrapper
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage

# PyPi Imports
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, root_validator




class CustomAgentExecutor(AgentExecutor):
    """ Our simple agent that will override the langchain executor implementation on key methods

    Attributes:
        agent (Union[BaseSingleActionAgent, BaseMultiActionAgent]):
            – The agent to run for creating a plan and determining actions to take at each step of the execution loop.
        tools (Sequence[BaseTool]):
            – The valid tools the agent can call.
        return_intermediate_steps (Optional[bool]):
            – Whether to return the agent's trajectory of intermediate steps at the end in addition to the final output.
        max_iterations (Optional[int]):
            – The maximum number of steps to take before ending the execution loop.
            – Setting to 'None' could lead to an infinite loop.
        max_execution_time (Optional[float]):
            – The maximum amount of wall clock time to spend in the execution loop.
        early_stopping_method (Optional[str]):
            – The method to use for early stopping if the agent never returns `AgentFinish`.
            – Either 'force' or 'generate'.
                - `"force"` returns a string saying that it stopped because it met a time or iteration limit.
                - `"generate"` calls the agent's LLM Chain one final time to generate a final answer based on the previous steps.
        handle_parsing_errors (Union[bool, str, Callable[[OutputParserException], str]]):
            – How to handle errors raised by the agent's output parser.
            – Defaults to `False`, which raises the error.
            – If `true`, the error will be sent back to the LLM as an observation.
            – If a string, the string itself will be sent to the LLM as an observation.
            – If a callable function, the function will be called with the exception as an argument, and the result of that function will be passed to the agent as an observation.
        trim_intermediate_steps (Union[int, Callable[[List[Tuple[AgentAction, str]]], List[Tuple[AgentAction, str]]]]):
            – How many intermediate steps to keep.
            – If negative, keeps all.
            – If positive, keeps the last n.
            – If callable, calls the function with the intermediate steps and uses the result.
    """
    def _return(
            self,
            output: AgentFinish,
            intermediate_steps: list,
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    # async def _areturn(self, ...)
    #    ... ASYNC TBD LATER ...

    def _take_next_step(
            self,
            name_to_tool_map: Dict[str, BaseTool],
            color_mapping: Dict[str, str],
            inputs: Dict[str, str],
            intermediate_steps: List[Tuple[AgentAction, str]],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """
        Take a single step in the thought-action-observation loop.

        This method controls how the agent makes and acts on choices within the
        decision-making process. It performs the following:
            1. Prepares the intermediate steps.
            2. Calls the agent to plan its next action(s).
            3. Handles OutputParserException if it occurs during planning.
            4. Checks if the output signals the end of the run.
                - If the chosen tool is a finishing tool, ends and returns AgentFinish.
            5. Looks up the tool and its details
            6. Calls the tool with the tool input to get an observation.
            7. Adds the action and resulting observation to the intermediate steps
            8. Returns the result.

        Args:
            name_to_tool_map (Dict[str, BaseTool]): A mapping of tool names to tool objects.
            color_mapping (Dict[str, str]): A mapping of tools to colors, used for logging.
            inputs (Dict[str, str]): Inputs to the agent's planning function.
            intermediate_steps (List[Tuple[AgentAction, str]]): The history of actions and observations.
            run_manager (Optional[CallbackManagerForChainRun]): A manager for callbacks during execution.

        Returns:
            Union[AgentFinish, List[Tuple[AgentAction, str]]]: Either an AgentFinish object if the agent is done or a list of action-observation tuples.
        """
        try:
            # Step 1: Prepare the intermediate steps
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Step 2: Call the agent's planning function to decide the next action
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            # Step 3: Handle exceptions raised during planning
            # Determine whether to raise the error or handle it
            raise_error = not self.handle_parsing_errors if isinstance(self.handle_parsing_errors, bool) else False
            if raise_error:
                raise e

            # Process the exception and obtain an observation
            observation = self._handle_parsing_error(e)

            # Create an AgentAction for the exception and run it
            output = AgentAction("_Exception", observation, str(e))
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]

        # Step 4: Check if the output signals the end of the run
        if isinstance(output, AgentFinish):
            return output

        actions = [output] if isinstance(output, AgentAction) else output
        result = []
        for agent_action in actions:
            # Step 5: Lookup the tool and its details
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            tool, return_direct, color, tool_run_kwargs = self._get_tool_details(
                agent_action, name_to_tool_map, color_mapping
            )

            # Step 6: Call the tool with the tool input to get an observation
            observation = tool.run(
                agent_action.tool_input,
                verbose=self.verbose,
                color=color,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )

            # Step 7: Add the action and observation to the result
            result.append((agent_action, observation))

        # Step 8: Return the result
        return result

    # async def _atake_next_step(self, ...)
    #    ... ASYNC TBD LATER ...

    def _call(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the agent's decision-making process and obtain a response.

        This method constructs necessary mappings for tools and color-coding (for logging),
        initializes counters for iterations and time tracking, and then enters a loop to
        repeatedly take next steps based on the agent's actions until a stopping condition is met.

        Args:
            inputs (Dict[str, str]): The inputs required by the agent to make decisions.
            run_manager (Optional[CallbackManagerForChainRun]): Manager for callback functions during the run.

        Returns:
            Dict[str, Any]: The agent's response, either the final decision or an indication that the process was stopped early.
        """

        # Step 1: Create a mapping of tool names to tool objects for later look-up
        name_to_tool_map = {tool.name: tool for tool in self.tools}

        # Step 2: Create a color mapping for tools, which is used for logging purposes
        color_mapping = get_color_mapping([tool.name for tool in self.tools], excluded_colors=["green", "red"])

        # Step 3: Initialize variables to keep track of intermediate steps, iterations, and time
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()

        # Step 4: Enter the agent loop, continuously taking the next step until a stopping condition is met
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
                inputs=inputs,
                intermediate_steps=intermediate_steps,
                run_manager=run_manager,
            )

            # Step 5: Check if the output signals the end of the process
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    output=next_step_output,
                    intermediate_steps=intermediate_steps,
                    run_manager=run_manager
                )

            # Step 6: Extend intermediate steps with the output from the next step
            intermediate_steps.extend(next_step_output)

            # Step 7: Check if there's a direct return from the tool
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        output=tool_return,
                        intermediate_steps=intermediate_steps,
                        run_manager=run_manager
                    )

            # Step 8: Update iterations and time elapsed
            iterations += 1
            time_elapsed = time.time() - start_time

        # Step 9: If the loop ended without returning, generate a stopped response and return it
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    # async def _acall(self, ...)
    #    ... ASYNC TBD LATER ...

    def _get_tool_return(
            self, next_step_output: Tuple[AgentAction, str]
    ) -> Optional[AgentFinish]:
        """
        Check if the given tool is configured to return a direct response.

        This method looks up the tool associated with the given agent action and checks if it is marked to return directly.
        If it is, an `AgentFinish` object is created and returned with the observation. If the tool is not found or not
        marked to return directly, the method returns None.

        Args:
            next_step_output (Tuple[AgentAction, str]): A tuple containing the agent's action and the observation associated with it.

        Returns:
            Optional[AgentFinish]: An `AgentFinish` object if the tool is marked to return directly, otherwise None.
        """
        # Step 1: Unpack the agent action and observation from the provided tuple
        agent_action, observation = next_step_output

        # Step 2: Create a mapping of tool names to tool objects for later look-up
        name_to_tool_map = {tool.name: tool for tool in self.tools}

        # Step 3: Check if the agent's action tool exists in the mapping (invalid tools won't be in the map)
        if agent_action.tool in name_to_tool_map:
            # Step 4: If the tool is found and is marked to return directly, create and return an `AgentFinish` object
            if name_to_tool_map[agent_action.tool].return_direct:
                return AgentFinish(return_values={self.agent.return_values[0]: observation}, log="")

        # Step 5: If the tool is not found or not marked to return directly, we will return None (obviously unnecessary)
        return None

    def _handle_parsing_error(self, e: OutputParserException) -> str:
        """ Handle the OutputParserException based on the `handle_parsing_errors` attribute.

        Args:
            e (OutputParserException): The exception to be handled.

        Returns:
            str: The resulting observation based on how the error is handled.
        """
        # Determine whether to raise the error or handle it
        if isinstance(self.handle_parsing_errors, bool):
            if e.send_to_llm:
                return str(e.observation)
            else:
                return "Invalid or incomplete response"
        elif isinstance(self.handle_parsing_errors, str):
            return self.handle_parsing_errors
        elif callable(self.handle_parsing_errors):
            return self.handle_parsing_errors(e)
        else:
            raise ValueError("Got unexpected type of `handle_parsing_errors`")

    def _get_tool_details(
            self,
            agent_action: AgentAction,
            name_to_tool_map: Dict[str, BaseTool],
            color_mapping: Dict[str, str]
    ) -> Tuple[Union[InvalidTool, BaseTool], bool, Union[str, None], Dict]:
        """
        Get the tool details for a given agent action.

        Args:
            agent_action (AgentAction): The agent action for which the tool is to be looked up.
            name_to_tool_map (Dict[str, BaseTool]): A mapping of tool names to tool objects.
            color_mapping (Dict[str, str]): A mapping of tools to colors, used for logging.

        Returns:
            Tuple[Union[BaseTool, InvalidTool], bool, Union[str, None], Dict]:
                - the tool object
                - whether the tool returns directly
                - its associated color,
                - additional logging arguments.
        """

        # Lookup the tool using the agent action's tool name and get its details
        if agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            return tool, return_direct, color, tool_run_kwargs
        else:
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            return InvalidTool(), False, None, tool_run_kwargs

    def _prepare_intermediate_steps(
            self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[Tuple[AgentAction, str]]:
        """ Prepares and returns the intermediate steps based on the 'trim_intermediate_steps' attribute.

        Args:
            intermediate_steps (List[Tuple[AgentAction, str]]):
                - A list of tuples containing AgentAction and corresponding descriptions.

        Returns (List[Tuple[AgentAction, str]]):
            - A list of tuples containing AgentAction and corresponding descriptions
        """

        # Check if 'trim_intermediate_steps' is an integer and greater than 0.
        # If it is, return the last 'trim_intermediate_steps' elements from 'intermediate_steps'.
        if (isinstance(self.trim_intermediate_steps, int)) and (self.trim_intermediate_steps > 0):
            return intermediate_steps[-self.trim_intermediate_steps:]

        # Check if 'trim_intermediate_steps' is a callable function.
        # If it is, call it with 'intermediate_steps' as an argument and return the result.
        elif callable(self.trim_intermediate_steps):
            return self.trim_intermediate_steps(intermediate_steps)

        # If 'trim_intermediate_steps' is neither an integer greater than 0 nor a callable,
        # return 'intermediate_steps' as it is.
        else:
            return intermediate_steps

def SimpleAgencyAgent(CustomAgentExecutor):
    """Take a single step in the thought-action-observation loop.

    Args:
        name_to_tool_map (Dict[str, BaseTool]): A mapping of tool names to tool objects.
        color_mapping (Dict[str, str]): A mapping of tools to colors, used for logging.
        inputs (Dict[str, str]): Inputs to the agent's planning function.
        intermediate_steps (List[Tuple[AgentAction, str]]): The history of actions and observations.
        run_manager (Optional[CallbackManagerForChainRun]): A manager for callbacks during execution.

    Returns:
        Union[AgentFinish, List[Tuple[AgentAction, str]]]: Either an AgentFinish object if the agent is done or a list of action-observation tuples.
    """
    pass


def SimpleAgent():
    # Set up a prompt template
    pass


class CustomOutputParser(AgentOutputParser):
    """
    A custom parser for interpreting the output of an LLM system.
    This class is a child of the `AgentOutputParser` class.

    methods:
        parse(llm_output: str) -> Union[AgentAction, AgentFinish]:
            - Processes a string output from an LLM system and returns
              an AgentAction or AgentFinish object based on the output content.
    """

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        Parses the given LLM output and returns an AgentAction or AgentFinish object.

        Args:
            llm_output (str)
                - The output string from the LLM system.

        Returns:
            - The AgentAction or AgentFinish object based on the LLM output content.
            - The returned value will be of type Union[AgentAction, AgentFinish]

        Raises:
            ValueError
                - If the LLM output doesn't contain a recognizable action and action input
        """

        # Check for a 'Final Answer:' in the LLM output
        if "Final Answer:" in llm_output:
            # Extract everything after 'Final Answer:' and remove leading/trailing white spaces
            final_answer = llm_output.split("Final Answer:")[-1].strip()

            return AgentFinish(
                # Return the final answer in a dictionary
                return_values={"output": final_answer},
                log=llm_output,
            )

        # Regular expression pattern for extracting the JSON blob between "Action:" and "```"
        regex = r"Action:\n```\n(.*?)\n```"

        # Use re.search() to find the match in the llm_output
        match = re.search(regex, llm_output, re.DOTALL)

        # If no action and action input are found, raise a ValueError
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        # Extract the JSON blob representing the action and action input
        json_blob = match.group(1).strip()

        # Parse the JSON blob to extract the action and action input
        json_data = json.loads(json_blob)
        action = json_data["action"].strip()
        action_input = json_data["action_input"].strip()

        # Return the action and action input in an AgentAction object
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


class CustomPromptTemplate(BaseChatPromptTemplate):

    # The template to use
    template: str

    # The list of tools available
    tools: List[Tool]

    # Other
    intermediary_step_str: str = "intermediate_steps"
    observation_str: str = "Observation: "
    post_observation_str: str = "Thought: "

    def format_messages(self, **kwargs):
        # Get the intermediate steps (AgentAction, Observation tuples)

        # Format them in a particular way
        thoughts, intermediate_steps = "", kwargs.pop(self.intermediary_step_str)
        for i, (action, observation) in enumerate(intermediate_steps):
            # thoughts += f"\nCYCLE {i+1}:\n"
            thoughts += action.log
            thoughts += f"\n{self.intermediary_step_str}: {observation}\n{self.post_observation_str}"

        # Set the agent_scratchpad variable to that value
        # print(f"\n\n\nTHOUGHTS:\n{'-'*50}", thoughts, f"\n{'-'*50}\n\n\n\n\n")
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # kwargs["vars"] = "\n".join([f"{_var.name}: {_var.description}" for _var in self.vars])

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # kwargs["var_names"] = ", ".join([_var.name for _var in self.vars])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


def get_simple_agent(chat_model, tools, model_name,
                     intermediary_step_str="intermediate_steps",
                     history_str = "chat_history",
                     human_str="HUMAN",
                     ai_str="ASSISTANT",
                     input_str = "input",
                     observation_strs=("\nObservation: ", "\nObservation: ", "\n\tObservation:")):

    prompt_name = "simple_agent_v2" if "openai" in model_name else "simple_agent_llama_v1"
    prompt = CustomPromptTemplate(
        template=prompt_library.get_prompt(prompt_name).template,
        tools=tools,
        # The history template includes "history" as an input variable so we can interpolate it into the prompt
        input_variables=[input_str, intermediary_step_str, history_str],
    )
    # prompt = prompt_library.get_prompt("simple_agent")

    # LLM chain consisting of the LLM and a prompt
    chat_chain = LLMChain(llm=chat_model.hf_pipe, prompt=prompt)
    agent = LLMSingleActionAgent(
        llm_chain=chat_chain,
        output_parser=StructuredChatOutputParserWithRetries(),
        stop=observation_strs,
        allowed_tools=[x.name for x in tools],
    )
    memory = ConversationBufferWindowMemory(k=5, memory_key=history_str, human_prefix=human_str, ai_prefix=ai_str)
    agent_executor = CustomAgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    return agent_executor

