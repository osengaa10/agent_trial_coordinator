from crewai import Agent
from tools import save_content, clinical_trials_search, human_tools 
from utils import print_agent_output, together_llm
# Define your agents with roles and goals
search_term_agent = Agent(
    role='Prompt Distiller',
    goal='Extract the main keyword from the human prompt describing their medical condition.',
    backstory="""As a top medical expert and and software engineer, you will review
    a patient's prompt and distill their prompt into a searchable word or phrase. 
    This simple search term will then be passed as a url parameter to an api.
    """,
    verbose=True,
    allow_delegation=False,
    # llm=ClaudeHaiku,
    llm=together_llm,
    max_iter=5,
    memory=False,
    step_callback=lambda x: print_agent_output(x,"===search_term_agent==="),
    tools=human_tools, # Passing human tools to the agent,
)

trials_search_agent = Agent(
    role='Clinical Trials Searcher',
    goal='Use extracted keywords to find relevant clinical trials',
    backstory="""As a top medical expert and and software engineer, you will use the key words
    from the Prompt Distiller to fetch a list of all relevant clinical trials.
    """,
    verbose=True,
    allow_delegation=False,
    # llm=ClaudeHaiku,
    llm=together_llm,
    max_iter=5,
    memory=True,
    step_callback=lambda x: print_agent_output(x,"===trials_search_agent==="),
    tools=[clinical_trials_search], # Passing human tools to the agent,
)

coordinator_agent = Agent(
    role='Clinical Trials Coordinator',
    goal='Review a list of clinical trials and make a list of the clinical trials that would be the best fit for the patient according to their prompt',
    backstory="""As a renowned Medical Expert you have an exceptional ability to determine the most
    promising clinical trials when provided with a description of the patient's health issue.
    Your ability to match patients with clinical trials has saved many lives. Your reviews are
    always comprehensive and thorough since you know lives depend on you.
    """,
    # llm=ClaudeHaiku,
    llm=together_llm,
    verbose=True,
    max_iter=5,
    memory=False,
    step_callback=lambda x: print_agent_output(x,"===coordinator_agent==="),
    allow_delegation=False,
    tools=human_tools
)

# archiver = Agent(
#     role='File Archiver',
#     goal='Take in information and write it to a Markdown file',
#     backstory="""You are a efficient and simple agent that gets data and saves it to a markdown file. in a quick and efficient manner""",
#     # llm=ClaudeHaiku,
#     llm=together_llm,
#     verbose=True,
#     step_callback=lambda x: print_agent_output(x,"Archiver Agent"),
#     tools=[save_content],
# )