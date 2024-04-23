import configs
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import utils

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# DEFAULT_SYSTEM_PROMPT = """You have 30 years of experience as both a professor of medicine and a highly esteemed researcher in human genetic engineering. Your goal is to invent novel treatments for human cancers.
# Always answer as helpfully as possible using the context text provided. 
# If a question does not make any sense, or is not factually coherent, provide what information is needed for the question to be answered. If you don't know the answer to a question, please don't share false information.
# Your superior logic and reasoning abilities coupled with you vast knowledge in biology, genetics, and medicine allow you to conduct innovative experiments resulting in significant advancements in medicine.
# """

DEFAULT_SYSTEM_PROMPT = """You have 30 years experience as both a practicing oncologist and a clinical trials coordinator.
Always try to find the clinical trials that best fit the patient using the context text provided. 
The clinical trials should contain the title, expected outcomes, contact information and description. 
If you don't know the answer to a question, please don't share false information.


"""

instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""



def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

get_prompt(instruction, DEFAULT_SYSTEM_PROMPT)

prompt_template = get_prompt(instruction, DEFAULT_SYSTEM_PROMPT)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question", "chat_history"]
)


chain_type_kwargs = {"prompt": llama_prompt}

memory = ConversationBufferMemory(memory_key="chat_history", input_key='query', output_key='result', return_messages=True)




def create_chain():
    """create the chain to answer questions"""
    print("===========CREATE_CHAIN===========")
    # PARAMETERIZED_SYSTEM_PROMPT = ''
    # if system_prompt == "":
    #     PARAMETERIZED_SYSTEM_PROMPT = """You have 30 years experience as both a practicing oncologist and a clinical trials coordinator.
    #     Always try to find the clinical trials that best fit the patient using the context text provided. 
    #     The clinical trials should contain the title, expected outcomes, contact information and description. 
    #     If you don't know the answer to a question, please don't share false information.
    #     """
    # else:
    #     PARAMETERIZED_SYSTEM_PROMPT = system_prompt

    PARAMETERIZED_SYSTEM_PROMPT = """You have 30 years experience as both a practicing oncologist and a clinical trials coordinator.
    When given a medical synopsis/report, find the clinical trials that best fit the patient. Always use the context text provided. 
    The clinical trials should contain the title, expected outcomes, contact information and description. 
    If you don't know the answer to a question, please don't share false information.
    """
    
    
    instruction = """CONTEXT:/n/n {context}/n

    Question: {question}"""
    SYSTEM_PROMPT = B_SYS + PARAMETERIZED_SYSTEM_PROMPT + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    # print(f"================create_user_chain system_prompt================")
    # print(prompt_template)
    # print(f"================create_user_chain system_prompt================")
    llama_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )
    persist_directory = f'db'
    chain_type_kwargs = {"prompt": llama_prompt}
    vectordb = Chroma(embedding_function=configs.embedding,persist_directory=persist_directory)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=utils.together_llm_rag,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        chain_type_kwargs=chain_type_kwargs,
                                        return_source_documents=True,
                                        verbose=True,
                                        memory=memory)