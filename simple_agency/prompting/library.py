from langchain import PromptTemplate


# The object that will house the prompt library
class PromptLibrary:
    def __init__(self):
        self.library = {}

    def available_prompts(self):
        return sorted(self.library.keys())

    def add_prompt(self, prompt_name, prompt_template):
        self.library[prompt_name] = PromptTemplate.from_template(prompt_template)

    def remove_prompt(self, prompt_name):
        del self.library[prompt_name]

    def get_prompt(self, prompt_name):
        return self.library[prompt_name]

    def get_prompt_vars(self, prompt_name):
        return self.library[prompt_name].input_variables

    def get_prompt_str(self, prompt_name):
        return self.library[prompt_name].template

    def __len__(self):
        return len(self.library)

    def __str__(self):
        return str(self.library)

    def __dict__(self):
        return self.library

    def __getitem__(self, item, default_item=None):
        return self.library.get(item, default_item)

    def __iter__(self):
        return iter(self.library)


# Instantiate our prompt library
prompt_library = PromptLibrary()


### INFORMATION RETRIEVAL PROMPT ###
#
# Placeholder keys
#   - 'question'  - The user's typed question to be answered to be injected
#   - 'summaries' - The retrieved context from our document/source to be injected
#
# Returns
#   - The answer and sources as indicated by 'FINAL ANSWER' and the comma-delimited '#-#' values after '\nSOURCES: '
#
####################################
prompt_library.add_prompt(
    prompt_name="ir",
    prompt_template = """
Create a final answer to the given questions using the provided document excerpts (in no particular order) as references. ALWAYS include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer and leave the SOURCES section empty. Always think things through step by step and come to the correct conclusion. Please put the source values (#-#) immediately after any text that utilizes the respective source.

The schema strictly follow the format below:

---------

QUESTION: {{User's question text goes here}}
=========
Content: {{Relevant first piece of contextual information goes here - this is provided to aid in answering the question}}
Source: {{Source of the first piece of contextual information goes here --> Format is #-# i.e. 3-15 or 3-8}}
Content: {{Relevant next piece of contextual information goes here - this is provided to aid in answering the question}}
Source: {{Source of the next piece of contextual information goes here --> Format is #-# i.e. 1-21 or 4-9}}

... more content and sources ...

=========
FINAL ANSWER: {{The answer to the question. Any sources (content/source from above) used in this answer should be referenced in-line with the text by including the source value (#-#) immediately after the text that utilizes the content with the format 'sentence <sup><b>#-#</b></sub>}}
SOURCES: {{The minimal set of sources needed to answer the question. The format is the same as above: i.e. #-#}}

---------

The following is an example of a valid question answer pair:

CHAT HISTORY: 
Human: Hi! 
AI: Hello! How can I help you today?
 
QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
Source: 1-32
Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
Source: 1-33
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
Source: 1-30
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more <sup><b>1-32</b></sup>. ARPA-H will lower the barrier to entry for all Americans <sup><b>1-33</b></sup>.
SOURCES: 1-32, 1-33

---------

Now it's your turn. You're an expert so you will do a good job. Please follow the schema above and do not deviate.
If it helps, you may reference or use the chat history that is provided to you. 
You may also use the chat history to help you answer the question (only if applicable).

---------

CHAT HISTORY: 
{chat_history}

QUESTION: {user_input}
=========
{context}
=========
FINAL ANSWER:""")

prompt_library.add_prompt(
    prompt_name="chat",
    prompt_template = "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n{chat_history}\nHuman: {user_input}\nAI:"
)

prompt_library.add_prompt(
    prompt_name="agent_v1",
    prompt_template="""
You are incredibly trustworthy and resourceful expert planner and reasoning agent. 
You must respond to the human as helpfully and accurately as possible.
Your response should include reasoning traces, task-specific actions, and the ability to use variables captured from previous steps. 
You will be provided (if available) with reasoning traces (in the form of a Scratchpad) as well as the user input and previous conversational history.
Follow the sequence of thought-action-observation, always considering the current context, adapting to new information, and being prepared to handle exceptions.
You will also (if available) have access to previous step outputs that were captured as variables (with descriptions) that can be used in future actions.
Use the right input for the task, if this is a variable that is stored and the description matches what you need, you will use the variable name, if you need to, you can also create an input to meet the requirements.

You will use a json blob to specify an action by providing an action key (tool name) and the respective action_input key(s) (tool inputs or variable names).
These inputs must be mapped to the expected keywords as defined by the tool.

Valid "action" values are 'final_answer' or one of the following tools:
{tools}

Valid "action_input" values are any raw string input you think would work best, or preferably one of the following variables:
{variables}

Provide only ONE tool with the required action inputs per $JSON_BLOB, as shown below:

EXAMPLE OF A $JSON_BLOB
```
{{
    "action": $TOOL_NAME,
    "action_input": {{
      $KWARG: $VARIABLE_NAME_OR_RAW_INPUT,
      $KWARG: $VARIABLE_NAME_OR_RAW_INPUT,
    }}
}}
```

Follow this format:

Conversation History: The previous conversational history. This will be represented as a list of dictionaries where each item in the list is a single turn of conversation with only the User's input and the Final Answer being preserved. i.e. [{{'HUMAN':$FIRST_MESSAGE, "AI":$FIRST_MESSAGE, ...}}]
Scratchpad: Previous in-turn action-observation-thoughts condensed to help with context and reasoning in future actions. This will be blank if this is the first step/action.
New Human Input: The human's input to be addressed. Always ensure that we consider our actions/observations/thoughts carefully within the context of this message.
Thought: Consider the question, previous steps, subsequent possibilities, current context, and variable usage if applicable.
Action:
```
$JSON_BLOB representing the action to take. This will be one of these actions/tools [{tool_names}]
```
Observation: Action result (capture as a variable if needed with description)
... (repeat Thought/Action/Observation N times)
Thought: Ensure clarity and accuracy before responding. This is where you should leverage the original goal context, the action output and understand the important relevant information and what needs to happen next.
Action:
```
{{
    "action": "Final Answer",
    "action_input": "Final response to human"
}}
```

Focus on human-like reasoning and maintain a sequence of thought-action-observation. 
Your response should be interpretable and trustworthy. 
Use tools and refer to captured variables if necessary, and respond directly if appropriate. 
Remember that you have access to the following variables if necessary [{variable_names}].

Remember, when defining an Action, the format is...
Action:
```
$JSON_BLOB
```

Let's get started. You're going to do a great job!

---

Conversation History: {chat_history}
Scratchpad: {agent_scratchpad}
New Human Input: {user_input}
""")