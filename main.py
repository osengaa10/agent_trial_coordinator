from crewai import Crew, Process
from agents import search_term_agent, coordinator_agent 
# archiver
from tasks import get_search_term, get_relevant_clinical_trials, curate_the_list
# saving_the_output
from utils import print_agent_output



# Instantiate your crew with a sequential process
crew = Crew(
    agents=[search_term_agent, coordinator_agent, 
            # archiver
            ],
    tasks=[get_search_term,
           get_relevant_clinical_trials,
           curate_the_list,
        #    saving_the_output
           ],
    verbose=2,
    process=Process.sequential,
    full_output=True,
    share_crew=False,
    step_callback=lambda x: print_agent_output(x,"MasterCrew Agent")
)

# Kick off the crew's work
results = crew.kickoff()

# Print the results
print("Crew Work Results:")
print(results)