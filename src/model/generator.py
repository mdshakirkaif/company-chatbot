# generator.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.model.loader import get_llm

def build_generation_chain(repo_id: str, prefer_local: bool = False):
    """
    Assembles an LCEL pipeline mapping input variables into the fine-tuned Unsloth prompt format.
    Includes an explicit string conversion block to safely interface with custom runnables.
    """
    # Grab our API-routed Lambda execution component
    llm = get_llm(repo_id, prefer_local=prefer_local)
    
    # Strictly mirrors your training prompt architecture
    prompt_template = "### Instruction:\n{question}\n\n### Response:\n"
    
    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template
    )
    
    # CRITICAL FIX: Add `lambda x: x.to_string()` right after the prompt
    # This unpacks StringPromptValue into a plain Python string that JSON can serialize.
    chain = prompt | (lambda x: x.to_string()) | llm | StrOutputParser()
    return chain