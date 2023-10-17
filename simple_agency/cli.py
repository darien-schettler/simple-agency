# Builtin Imports
import argparse

# Install imports
from langchain.agents import load_tools
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain.tools import StructuredTool

# Local package imports
from simple_agency.config.authentication import return_infra_keys
from simple_agency.model_manager.model_loader import chat_model_from_name
from simple_agency.agent.openai import get_simple_agent
from simple_agency.tools.utils import get_tools
from simple_agency.runhouse_ops.instance_handler import get_llama_agent


def parse_args():
    parser = argparse.ArgumentParser(description='Command-line interface for the backend.')
    parser.add_argument('--debug', action='store_false', help='Enable debug mode')
    parser.add_argument('--name', '-n', type=str, default="gpt-4", help='The chat model we want to use')
    parser.add_argument('--temperature', '-t', type=float, default=0.6, help='The temperature of the model')
    parser.add_argument('--use_streaming', '-s', type=bool, default=True, help='Whether to use streaming or not')
    args = parser.parse_args()

    # Check if debug mode is enabled and if so print out useful info
    if args.debug:
        # langchain.debug = False
        print('\n... Debug mode enabled ...\n\n\t... Args ...')
        for k, v in vars(args).items(): print(f'\t{k:>15}: {v}')

    return args


_LLAMA=True
_RESTART=True


def main():
    # Parse the command line arguments
    args = parse_args()

    # Load the API keys and authenticate as needed
    infra_keys = return_infra_keys("paperspace")

    # Load the model
    tools = get_tools()

    if not _LLAMA:
        chat_model = chat_model_from_name(
            model_name=args.name,
            temperature=args.temperature,
            use_streaming=args.use_streaming
        )
        agent = get_simple_agent(chat_model, tools, args.name)
    else:
        agent = get_llama_agent(model_name='llama2-7b', tools=tools, force_rh_restart=_RESTART)

    while True:
        # Prompt the user for a document path
        user_input = input("\nEnter a task for our agent to perform and/or just chat (or press Enter to quit):\n\t>>> ")

        if not user_input:
            print('\n... Exiting ...\n')
            break

        agent.run(user_input)


if __name__ == '__main__':
    main()
