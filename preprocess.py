from urielplus.urielplus import URIELPlus
from pyglottolog import Glottolog
import numpy as np
import pandas as pd


def get_metadata(languages, output_path="metadata.csv"):
    """
    Get Glottolog metadata, and save it as a CSV file.
    :param languages: List[str], list of glottocodes (of languages covered by URIEL+)
    :param output_path: str, path to save the metadata CSV file
    :return:
    """
    # For each language in languages, get the Glottolog metadata (name, family, macroarea, etc.)
    # Save it as a CSV with index being Glottocodes and columns being metadata fields


def compute_correlation_matrix(df, output_path="correlation_matrix.csv"):
    """
    Compute the correlation matrix between different features in the dataframe, and save it as a CSV file.
    :param df: pd.DataFrame, typological dataset
    :param output_path: str, path to save the correlation matrix CSV file
    """
    # Make sure to account for missing values


def compute_genetic_neighbours(languages, k=5, output_path="genetic_neighbours.json"):
    """
    Compute the k nearest genetic neighbours using Glottolog.
    :param languages: List[str], list of glottocodes (of languages covered by URIEL+)
    :param k: int, number of nearest neighbours to retrieve
    :param output_path: str, path to save the genetic neighbours JSON file
    """
    # For each language in languages, retrieve its Glottolog tree
    # You can use glottolog.newick_tree(glottocode) to get the Newick tree for a language
    # Retrieve the k nearest genetic neighbours for each language
    # Save the dictionary as a JSON (key: glottocode, value: list of k nearest neighbours)


def compute_geographic_neighbours(languages, k=5, output_path="geographic_neighbours.json"):
    """
    Compute the k nearest geographic neighbours using Glottolog.
    :param languages: List[str], list of glottocodes (of languages covered by URIEL+)
    :param k: int, number of nearest neighbours to retrieve
    :param output_path: str, path to save the geographic neighbours JSON file
    """
    # For each language in languages, retrieve its longitude and latitude
    # The brute force strategy would be to compute the geographic distance between each pair of languages
    # then retrieve the k nearest geographic neighbours for each language
    # but feel free to use more efficient algorithms


if __name__ == '__main__':
    u = URIELPlus()
    input_path = ...  # Fill this with the path to the Glottolog data
    glottolog = Glottolog(input_path)
    # Construct the URIEL+ typological dataframe, with glottocodes as the index and typological features as columns
    # (No need to impute, but you will need to integrate databases.
    # You might want to repeat this process for both union aggregation and average aggregation)
    typ_df = ...

    # Compute and save the language metadata
    # Compute the correlation matrix and save it
    # Extract the list of languages (in glottocode) in URIEL+
    languages = ...
    # Compute the genetic neighbours and save it
    # Compute the geographic neighbours and save it
    typ_df.to_csv("uriel+_typological.csv") # Save the typological data
