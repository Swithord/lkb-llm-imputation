import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


def construct_prompt(language, feature) -> (str, str):
    """
    Construct the prompt for the given language and feature.
    :param language: str, Glottocode of the language to impute
    :param feature: str, the typological feature to impute
    :return: (str, str), the system and user prompts
    """
    system = ...

    user = ''
    # Part 1: Metadata

    # Part 2: Known typological features
    top_n = 10 # Or some other value
    # Using corr_df, extract the top_n features that have data for the given language

    # Part 3: Genetic neighbours
    # Add all the neighbours from genetic_neighbours

    # Part 4: Geographic neighbours
    # Add all the neighbours from geographic_neighbours

    return system, user


def run_llama(prompt) -> str:
    """
    Run the LLaMA model with the given prompt.
    :param prompt: str, the prompt to query the model
    :return:
    """
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')

    # Query the model with the prompt and return the output
    output = ''
    return output


if __name__ == '__main__':
    # Read the typological data
    typ_df = pd.read_csv(...)
    # Read the metadata file
    metadata_df = pd.read_csv(...)
    # Read the typological features correlation matrix
    corr_df = pd.read_csv(...)
    # Read the genetic neighbours json file
    with open(..., 'r') as f:
        genetic_neighbours = json.load(f)
    # Read the geographic neighbours json file
    with open(..., 'r') as f:
        geographic_neighbours = json.load(f)

    impute_values = {} # Define the (language, feature) pairs to impute
    # Construct prompts and query the model
    # Extract the classification from the output and save the results
