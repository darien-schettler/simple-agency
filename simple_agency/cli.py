# Builtin Imports
import os
import argparse

# Install imports
import langchain

# Local package imports
from simple_agency.config.authentication import return_infra_keys


def parse_args():
    parser = argparse.ArgumentParser(description='Command-line interface for the backend.')
    parser.add_argument('--debug', action='store_false', help='Enable debug mode')
    parser.add_argument('--name', '-n', type=str, default="gpt-4", help='The chat model we want to use')
    parser.add_argument('--temperature', '-t', type=float, default=0.6, help='The temperature of the model')
    parser.add_argument('--use_streaming', '-s', type=bool, default=True, help='Whether to use streaming or not')
    args = parser.parse_args()

    # Check if debug mode is enabled and if so print out useful info
    if args.debug:
        langchain.debug = True
        print('\n... Debug mode enabled ...\n\n\t... Args ...')
        for k, v in vars(args).items(): print(f'\t{k:>15}: {v}')

    return args


def main():
    # Parse the command line arguments
    args = parse_args()

    # Load the API keys and authenticate as needed
    infra_keys = return_infra_keys("paperspace")

    # Initialize the memory
    memory_store = []

    # Load the model
    chat_model = chat_model_from_name(
        model_name=args.name,
        temperature=args.temperature,
        use_streaming=args.use_streaming
    )
    chat_chain = prompt_library.get_prompt("chat") | chat_model
    doc_qa_chain = prompt_library.get_prompt("ir") | chat_model

    # Initilaize some placeholders and empty values
    available_vs = None
    response = ""


    # Start the chatbot
    while True:

        # Prompt the user for a document path
        user_input = input("\nEnter a question or path/name of document to parse (or press Enter to quit):\n\t>>> ")

        if not user_input:
            print('\n... Exiting ...\n')
            break

        file_names = [x for x in user_input.split() if x.endswith((".txt", ".pdf", ".doc"))]
        if len(file_names)>0:
            f_path = os.path.join(
                    "/Users/darienschettler/PycharmProjects/Boiler-LLM-App/docs/cli_testing",
                    file_names[-1]
                )
            available_vs = process_document(path=f_path)
            response = f"AI: I've retrieved the vector store for the file {file_names[-1]}. " \
                       f"I'm now ready to answer your questions with this new context ..."
            print(response.split("AI: ")[-1])

        elif user_input.lower().startswith("qa: "):
            if available_vs is not None:
                context = search_docs(available_vs, user_input, top_k=args.top_k)
                response = doc_qa_chain.run(
                    user_input=user_input, context=context, chat_history="\n".join(memory_store)
                )
            else:
                print("Please enter a document path first ...")
        else:
            response = chat_chain.run(user_input=user_input, chat_history="\n".join(memory_store))

        memory_store += [f"Human: {user_input}", f"AI: {response}"]


if __name__ == '__main__':
    main()
