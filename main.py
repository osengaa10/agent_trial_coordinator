from crewai import Crew, Process
from agents import search_term_agent, coordinator_agent, trials_search_agent, patient_consultant_agent
# archiver
from tasks import get_search_term, get_relevant_clinical_trials, curate_the_list, create_medical_report
# saving_the_output
from utils import print_agent_output
import json

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[search_term_agent, trials_search_agent, patient_consultant_agent, coordinator_agent, 
            # archiver
            ],
    tasks=[get_search_term,
           get_relevant_clinical_trials,
           create_medical_report,
           curate_the_list,
        #    saving_the_output
           ],
    verbose=4,
    process=Process.sequential,
    full_output=True,
    share_crew=False,
    step_callback=lambda x: print_agent_output(x,"MasterCrew Agent")
)

# Kick off the crew's work
results = crew.kickoff()

# def append_json_to_file(data, filename='output.txt'):
    # """Append a dictionary as JSON to a file."""
    # with open(filename, 'a') as file:
    #     json.dump(data, file)
    #     file.write('\n')  # Write a newline character after each JSON object


def write_data_to_file(data, filename='output.txt'):
    """Write data to a file, overwriting existing content."""
    with open(filename, 'a') as file:
        file.write(str(data))
        file.write("\n")
# Print the results
print("Crew Work Results:")
print(results)



# def write_dict_to_file(data, filename='output_dict.txt'):
#     """Write dictionary data to a file in JSON format, appending to existing content."""
#     with open(filename, 'a') as file:  # Use 'a' to append; change to 'w' if overwriting is needed
#         # Convert the dictionary to a JSON-formatted string with indentation for readability
#         formatted_data = json.dumps(data, indent=4)
#         file.write(formatted_data)
#         file.write("\n")

write_data_to_file(results)
# write_dict_to_file(results)