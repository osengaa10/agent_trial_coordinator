from datetime import datetime
from random import randint
import json
import os
import requests
from langchain.tools import tool
from crewai import Agent, Task, Crew, Process
# from crewai_tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools


@tool("save_content")
def save_content(task_output):
    """Useful to save content to a markdown file. Input is a string"""
    print('in the save markdown tool')
    # Get today's date in the format YYYY-MM-DD
    today_date = datetime.now().strftime('%Y-%m-%d')
    # Set the filename with today's date
    filename = f"{today_date}_{randint(0,100)}.md"
    # Write the task output to the markdown file
    with open(filename, 'w') as file:
        file.write(task_output)
        # file.write(task_output.result)

    print(f"Blog post saved as {filename}")

    return f"Blog post saved as {filename}, please tell the user we are finished"



search_tool = DuckDuckGoSearchRun()

# Loading Human Tools
human_tools = load_tools(["human"])





@tool("clinical_trials_search")
def clinical_trials_search(condition: str) -> str:
    """Fetches data from ClinicalTrials.gov API for a given condition. Input is a medical condition search term"""  
    # Base URL for the ClinicalTrials.gov API
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    # Parameters for the API request
    params = {
        "format": "json",
        "markupFormat": "markdown",
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING"
    }
    
    # Make the GET request
    response = requests.get(base_url, params=params)
    raw_trials = response.json()
    
    study_details_list = []
    # for study in raw_trials["studies"]:
    #     protocol_section = study.get("protocolSection", {})
    #     study_details_list.append({
    #     "officialTitle": protocol_section.get("identificationModule", {}).get("officialTitle"),        
    #     "eligibilityModule": protocol_section.get("eligibilityModule"),
    #     "centralContacts": protocol_section.get("contactsLocationsModule", {}).get("centralContacts"),
    #     "conditions": protocol_section.get("conditionsModule", {}).get("conditions"),
    #     "armsInterventionsModule": protocol_section.get("armsInterventionsModule"),
    #     "outcomesModule": protocol_section.get("outcomesModule"),
    #     "designModule": protocol_section.get("designModule")
    # })
        
    for i in range(0, 2):
        protocol_section = raw_trials["studies"][i].get("protocolSection", {})
        study_details_list.append({
        "officialTitle": protocol_section.get("identificationModule", {}).get("officialTitle"),        
        "eligibilityModule": protocol_section.get("eligibilityModule"),
        "centralContacts": protocol_section.get("contactsLocationsModule", {}).get("centralContacts"),
        "conditions": protocol_section.get("conditionsModule", {}).get("conditions"),
        "armsInterventionsModule": protocol_section.get("armsInterventionsModule"),
        "outcomesModule": protocol_section.get("outcomesModule"),
        "designModule": protocol_section.get("designModule")
    })
    # Return the response in JSON format
    return study_details_list
