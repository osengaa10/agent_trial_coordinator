from datetime import datetime
from random import randint
import json
import os
import requests
from langchain.tools import tool
from crewai import Agent, Task, Crew, Process
from utils import create_trial_report, wrap_text_preserve_newlines, process_llm_response
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from rag_pipeline import chunk_and_embed
from crewai_tools import BaseTool
import retrieval

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
        "filter.overallStatus": "RECRUITING",
        "pageToken": None
    }
    
    # Make the GET request
    response = requests.get(base_url, params=params)
    raw_trials = response.json()
    
    study_details_list = []
    counter = 0
    while True:
        # Make the GET request
        response = requests.get(base_url, params=params)
        # print("response: ", response)
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}")
            break  # Break the loop if there's an error in the response

        raw_trials = response.json()
        # Extract studies and append details to the list
        for study in raw_trials["studies"]:
            protocol_section = study.get("protocolSection", {})
            study_details_list.append({
                "officialTitle": protocol_section.get("identificationModule", {}).get("officialTitle"),        
                "eligibilityModule": protocol_section.get("eligibilityModule"),
                "centralContacts": protocol_section.get("contactsLocationsModule", {}).get("centralContacts"),
                "conditions": protocol_section.get("conditionsModule", {}).get("conditions"),
                "armsInterventionsModule": protocol_section.get("armsInterventionsModule"),
                "outcomesModule": protocol_section.get("outcomesModule"),
                "designModule": protocol_section.get("designModule")
            })

        # Update the pageToken to the next page token from the response, if any
        nextPageToken = raw_trials.get("nextPageToken")
        if not nextPageToken or counter > 7:
            break
        params["pageToken"] = nextPageToken
        counter = counter + 1

    print("Number of studies found:", len(study_details_list))
    for index, trial in enumerate(study_details_list):
        report_content = create_trial_report(trial)
        with open(f"./studies/clinical_trial_report_{index + 1}.txt", "w") as file:
            file.write(report_content)
    chunk_and_embed()
    # Return the response in JSON format
    return "Clinical trials search completed."





    
@tool("chromadb_retrieval_tool")
def chromadb_retrieval_tool(query: str) -> str:
    """Fetches answers from a chromaDB vector store using LangChain's RetrievalQA chain."""
    # query = "what are the best clinical trials for a patient with sarcoma that started in the stomach and has spread to the liver?"
    print("QUERY FROM CONSULTANT::::", query)
    qa_chain = retrieval.create_chain()
    llm_response = qa_chain(query)
    print("RESPONSE FROM CHROMADB::::", llm_response)
    wrap_text_preserve_newlines(llm_response['result'])
    sources = []
    try:
        for source in llm_response["source_documents"]:
            sources.append(source)
    except:
        print("NO SOURCES??")
        pass
 
    return {
        "answer": process_llm_response(llm_response),
        "sources": sources
    }