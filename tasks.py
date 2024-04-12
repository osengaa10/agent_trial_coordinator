from datetime import datetime
from crewai import Task
from agents import search_term_agent, coordinator_agent, trials_search_agent
# Create tasks for your agents
get_search_term = Task(
  description=f"""ASK THE HUMAN for their mediacl diagnosis. Extract keywords from the human\'s medical condition description.""",
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

curate_the_list = Task(
  description="""Using the search results from the Clinical Trials Searcher,
  curate a list of the most relevant clinical trials to the patient. Review the eligibility
  requirements of the search results to determine best fit. If any additional context is needed to determine clinical trial eligibility,
  you may ask the human basic follow up questions about themselves or their condition. For each clinical trial in your list, you 
  will provide the official title, contact info and an explanation on how the patient would be a good fit for the clinical trial.""",
  expected_output="""A prioritized list of the best clinical trials for the human. Each clinical trial has contact info, a title, and description on what makes them a good fit.
  """,
  agent=coordinator_agent,
  context=[get_relevant_clinical_trials]
)


# saving_the_output = Task(
#   description="""Taking the post created by the writer, take this and save it to a markdown file.
#   Your final answer MUST be a response must be showing that the file was saved .""",
#   expected_output='A saved file name',
#   agent=archiver,
#   context=[get_relevant_clinical_trials]
# )

