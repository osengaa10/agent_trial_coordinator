from dotenv import load_dotenv
import os
import json
from typing import Union, List, Tuple, Dict
from langchain.schema import AgentFinish
from langchain_core.agents import AgentFinish
from langchain_anthropic import ChatAnthropic
import together
from typing import Any, Dict
from pydantic import Extra
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from langchain.llms import Together
import textwrap

load_dotenv()

os.environ['TOGETHER_API_KEY']
os.environ['ANTHROPIC_API_KEY']
together.api_key = os.environ["TOGETHER_API_KEY"]

agent_finishes  = []

call_number = 0



def create_trial_report(trial):
    """Generate a formatted string for a clinical trial report."""
    # Safely access values, replacing None with 'Not available'
    title = trial.get('officialTitle', 'Not available')
    eligibility_criteria = trial['eligibilityModule'].get('eligibilityCriteria', 'Not available')
    
    report = f"Title: {title}\n\n"
    report += "Eligibility Criteria:\n"
    report += eligibility_criteria + "\n\n"
    
    report += "Contact Information:\n"
    # Ensure centralContacts is iterable, default to empty list if None
    for contact in trial.get('centralContacts', []) or []:
        name = contact.get('name', 'Not available')
        role = contact.get('role', 'Not available')
        phone = contact.get('phone', 'Not available')
        email = contact.get('email', 'Not available')
        report += f"{name} - {role}\n"
        report += f"Phone: {phone}, Email: {email}\n\n"
    
    # Ensure conditions is iterable, default to empty list if None
    conditions = ", ".join([condition for condition in trial.get('conditions', []) if condition])
    report += "Conditions:\n" + conditions + "\n\n"
    
    report += "Interventions:\n"
    # Check if 'armsInterventionsModule' exists and is not None before accessing 'interventions'
    interventions = (trial.get('armsInterventionsModule', {}) or {}).get('interventions', [])
    for intervention in interventions:
        intervention_type = intervention.get('type', 'Not available')
        intervention_name = intervention.get('name', 'Not available')
        description = intervention.get('description', 'Not available')
        report += f"{intervention_type} - {intervention_name}: {description}\n"
    
    report += "\nPrimary Outcomes:\n"
    # Ensure primaryOutcomes is iterable, default to empty list if None
    for outcome in trial['outcomesModule'].get('primaryOutcomes', []) or []:
        measure = outcome.get('measure', 'Not available')
        description = outcome.get('description', 'Not available')
        time_frame = outcome.get('timeFrame', 'Not available')
        report += f"{measure}: {description} (Time frame: {time_frame})\n"
    
    return report






def print_agent_output(agent_output: Union[str, List[Tuple[Dict, str]], AgentFinish], agent_name: str = 'Generic call'):
    global call_number  # Declare call_number as a global variable
    call_number += 1
    with open("crew_callback_logs.txt", "a") as log_file:
        # Try to parse the output if it is a JSON string
        if isinstance(agent_output, str):
            try:
                agent_output = json.loads(agent_output)  # Attempt to parse the JSON string
            except json.JSONDecodeError:
                pass  # If there's an error, leave agent_output as is

        # Check if the output is a list of tuples as in the first case
        if isinstance(agent_output, list) and all(isinstance(item, tuple) for item in agent_output):
            print(f"-{call_number}----Dict------------------------------------------", file=log_file)
            for action, description in agent_output:
                # Print attributes based on assumed structure
                print(f"Agent Name: {agent_name}", file=log_file)
                print(f"Tool used: {getattr(action, 'tool', 'Unknown')}", file=log_file)
                print(f"Tool input: {getattr(action, 'tool_input', 'Unknown')}", file=log_file)
                print(f"Action log: {getattr(action, 'log', 'Unknown')}", file=log_file)
                print(f"Description: {description}", file=log_file)
                print("--------------------------------------------------", file=log_file)

        # Check if the output is a dictionary as in the second case
        elif isinstance(agent_output, AgentFinish):
            print(f"-{call_number}----AgentFinish---------------------------------------", file=log_file)
            print(f"Agent Name: {agent_name}", file=log_file)
            agent_finishes.append(agent_output)
            # Extracting 'output' and 'log' from the nested 'return_values' if they exist
            output = agent_output.return_values
            # log = agent_output.get('log', 'No log available')
            print(f"AgentFinish Output: {output['output']}", file=log_file)
            # print(f"Log: {log}", file=log_file)
            # print(f"AgentFinish: {agent_output}", file=log_file)
            print("--------------------------------------------------", file=log_file)

        # Handle unexpected formats
        else:
            # If the format is unknown, print out the input directly
            print(f"-{call_number}-Unknown format of agent_output:", file=log_file)
            print(type(agent_output), file=log_file)
            print(agent_output, file=log_file)



class TogetherLLM(LLM):
    """Together large language models."""
    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""
    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""
    class Config:
        extra = Extra.forbid
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"
    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        print("output:::::", output)
        text = output['output']['choices'][0]['text']
        return text





together_llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.4,
    max_tokens=15000,
    top_k=1,
    # together_api_key="..."
)

together_llm_rag = TogetherLLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature = 0.1,
    max_tokens = 15000,
)





ClaudeHaiku = ChatAnthropic(
    model="claude-3-haiku-20240307"
)

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    return llm_response['result']