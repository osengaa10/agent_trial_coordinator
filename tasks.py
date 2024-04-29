from datetime import datetime
from crewai import Task
from agents import search_term_agent, coordinator_agent, trials_search_agent, patient_consultant_agent
from tools import chromadb_retrieval_tool, human_tools
# Create tasks for your agents
get_search_term = Task(
  description=f"""ASK THE HUMAN for their medical diagnosis. Extract keywords from the human\'s medical condition description.""",
  expected_output='A keyword or set of keywords',
  agent=search_term_agent,

)

get_relevant_clinical_trials = Task(
  description=f"""Search for clinical trials using the extracted keywords.
  """,
  expected_output='A comprehensive json object containing all relevant clinical trials to the humans condition, leave nothing out',
  agent=trials_search_agent,
  context=[get_search_term]
)

create_medical_report = Task(
    description='Converse with the human to gather information about their medical condition. Use this information to create a medical report for the Clinical Trials Coordinator. Ask the human as many questions as needed to build an informative report.',
    expected_output="""A short medical report containing the patient's medical condition and any relevant information for the Clinical Trials Coordinator.
    """,
    tools=human_tools,
    agent=patient_consultant_agent
)


# curate_the_list = Task(
#   description="""Using the search results from the Clinical Trials Searcher,
#   curate a list of the most relevant clinical trials to the patient. Review the eligibility
#   requirements of the search results to determine best fit. If any additional context is needed to determine clinical trial eligibility,
#   you may ask the human basic follow up questions about themselves or their condition. For each clinical trial in your list, you 
#   will provide the official title, contact info and an explanation on how the patient would be a good fit for the clinical trial.""",
#   expected_output="""A prioritized list of the best clinical trials for the human. Each clinical trial has contact info, a title, and description on what makes them a good fit.
#   """,
#   agent=coordinator_agent,
#   context=[get_relevant_clinical_trials]
# )


# retrieval_tool = ChromaDBRetrievalTool()

curate_the_list = Task(
    # description="""Using the context provided in the chromaDB vector store,
    # curate a list of the most relevant clinical trials to the patient. Review the eligibility
    # requirements of the search results to determine best fit. If any additional context is needed to determine clinical trial eligibility,
    # you may ask the human basic follow up questions about themselves or their condition. For each clinical trial in your list, you 
    # will provide the official title, contact info and an explanation on how the patient would be a good fit for the clinical trial.
    # """,
    description='Retrieve best clinical trials for patient.',
    expected_output="""A prioritized list of the best clinical trials for the human. Each clinical trial has contact info, a title, description on what makes them a good fit
    and what the risks and benefits are.
    """,
    tools=[chromadb_retrieval_tool],
    context=[create_medical_report],
    agent=coordinator_agent
)


# saving_the_output = Task(
#   description="""Taking the post created by the writer, take this and save it to a markdown file.
#   Your final answer MUST be a response must be showing that the file was saved .""",
#   expected_output='A saved file name',
#   agent=archiver,
#   context=[get_relevant_clinical_trials]
# )

