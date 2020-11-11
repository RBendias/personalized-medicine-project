# -*- coding: utf-8 -*-
# +
import numpy as np
import math
import re
import pandas as pd
import os

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw
from sklearn.base import BaseEstimator, TransformerMixin

data_path = "../../data/msk-redefining-cancer-treatment"

grantham_distances = pd.read_csv(
    os.path.join(data_path, "external/physiochem.csv"), sep=","
)

pc5 = {
    "I": "A",  # Aliphatic
    "V": "A",
    "L": "A",
    "F": "R",  # Aromatic
    "Y": "R",
    "W": "R",
    "H": "R",
    "K": "C",  # Charged
    "R": "C",
    "D": "C",
    "E": "C",
    "G": "T",  # Tiny
    "A": "T",
    "C": "T",
    "S": "T",
    "T": "D",  # Diverse
    "M": "D",
    "Q": "D",
    "N": "D",
    "P": "D",
}

def find_distance(AA1=None, AA2=None):
    if AA1 in pc5 and AA2 in pc5 and AA1 != "W":
        AAlist = grantham_distances.loc[
            grantham_distances["FIRST"] == AA1
        ]  # Finds row for AA1
    else:
        return float("nan")
    if AA2 == "S" or AA1 == "W":
        dist = np.NaN
    else:
        dist = AAlist.get(AA2)  # Search for AA2
    if math.isnan(dist):  # If not found, switch order and search again
        dist = find_distance(AA1=AA2, AA2=AA1)
    return int(dist)


def get_first_last_letter(variation):
    if re.search(r"^[A-Z]\d{1,7}[A-Z]$", variation):
        return variation[0], variation.split()[0][-1]
    else:
        return np.NaN, np.NaN


def get_phsysiochem_distance(variation):
    first_letter, last_letter = get_first_last_letter(variation)
    phsysiochem_distance = find_distance(first_letter, last_letter)
    return phsysiochem_distance


def get_data(text_file_path, variants_file_path, solution_file_path=None):
    variants_df = pd.read_csv(os.path.join(data_path, variants_file_path))
    text_df = pd.read_csv(
        os.path.join(data_path, text_file_path),
        sep="\|\|",
        engine="python",
        skiprows=1,
        names=["ID", "Text"],
    )

    merge_df = variants_df.merge(text_df, left_on="ID", right_on="ID")
    # Delete Samples with Empty Text
    merge_df = merge_df.loc[~merge_df.Text.isnull()]

    # Validation Solution
    if solution_file_path:
        class_df = pd.read_csv(os.path.join(data_path, solution_file_path))
        class_df.columns = ["ID", 1, 2, 3, 4, 5, 6, 7, 8, 9]
        class_df = (
            class_df.melt("ID", var_name="Class")
            .query("value== 1")
            .sort_values(["ID", "Class"])
            .drop("value", 1)
        )
        merge_df = merge_df.merge(class_df, left_on="ID", right_on="ID")

    return merge_df


class CustRegressionVals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = x.drop(["Gene", "Variation", "ID", "Text"], axis=1).values
        return x


class CustTxtCol(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.key].apply(str)


def extract_text_sections(text, gene, variation, section_length):
    """
    Given a variation name, text sections surrounding the variation in the text are extracted.
    
    """
    section = ""
    index = text.find(variation)
    if index == -1:
        variation = "mutation"
        index = text.find(variation)
    end_index = 0
    t = 0
    # print(variation,index,t)
    index_list = []
    while index != -1:
        index_list.append(index)
        old_index = index
        index = text.find(variation, old_index + 1)
        t += 1
    # for the case the variation appears less than 5 times
    if t < 5 and t > 0:
        for index in index_list:
            # if two sections are overlapping
            if index <= end_index:
                # determine end point depending on if it's out of range or not
                if index + int(2000 / t) <= (len(text) - 1):
                    end = index + int(2000 / t)
                else:
                    end = len(text) - 1
                section += text[end_index:end]
            else:
                # determine end point depending on if it's out of range or not
                if index + int(2000 / t) <= (len(text) - 1):
                    end = index + int(2000 / t)
                else:
                    end = len(text) - 1
                # determine start point depending on if it's out of range or not
                if index - int(2000 / t) >= 0:
                    start = index - int(2000 / t)
                else:
                    start = 0
                section += text[start:end]
            end_index = end
    if t == 0:
        section = text
    else:
        for index in index_list:
            if index <= end_index:
                # determine end point
                if index + section_length <= (len(text) - 1):
                    end = index + section_length
                else:
                    end = len(text) - 1
                section += text[end_index:end]
            else:
                if index + section_length <= (len(text) - 1):
                    end = index + section_length
                else:
                    end = len(text) - 1
                if index - section_length >= 0:
                    start = index - section_length
                else:
                    start = 0
                section += text[start:end]
            end_index = end
    section = section.replace("\t", " ")
    return section


# +
def custom_preprocessor(doc):
    return (
        doc.replace(",", " ")
        .replace("'", " ")
        .replace("*", " ")
        .replace(". ", " ")
        .replace("“", " ")
        .replace("”", " ")
        .replace("%", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace("\n", " ")
    )

def custom_tokenizer(doc):
    tokens = re.findall(r"(?=[^A-z]*[A-z][^A-z]*)(?=[\D]{3,}\s)(?=\b)\S+", doc)
    return tokens

# at least one letter, might contain hyphens, preceeded by white space. Second paranthesis makes sure we have a least three digits and/or letters
def tokenizer_at_least_three(doc):
    tokens = re.findall(
        r"(?=[^A-z]*[A-z][^A-z]*)(?=[A-z0-9]+-?[A-z0-9]+-?[A-z0-9]+\s)(?<=\s)\w\S+\w",
        doc,
    )
    return tokens

def tokenizer_no_numbers(doc):
    tokens = re.findall(r"(?=[^A-z]*[A-z][^A-z]*)(?=[^\d\s]{3,})(?<=\s)\w\S+\w", doc)
    return tokens

def tokenizer_only_letters(doc):
    tokens = re.findall(r"(?=[A-z]+-?[A-z]+-?[A-z]+\s)(?<=\s)\S{3,}", doc)
    return tokens

# takes at words with at least five letters
def tokenizer_five_letters(doc):
    tokens = re.findall(r"(?=[A-z]+-?[A-z]+-?[A-z]+\s)\S{5,}", doc)
    return tokens


# -

stop_words = [
    "fig",
    "figure",
    "et",
    "al",
    "table",
    "data",
    "analysis",
    "analyze",
    "study",
    "method",
    "result",
    "conclusion",
    "author",
    "find",
    "found",
    "show",
    "perform",
    "demonstrate",
    "evaluate",
    "discuss",
    "google",
    "scholar",
    "pubmed",
    "web",
    "science",
    "crossref",
    "supplementary",
    "(fig.)",
    "(figure",
    "fig.",
    "al.",
    "did",
    "thus,",
    "...",
    "interestingly,",
    "and/or",
    "author",
] + list(esw)

stop_words_extended = [
    "variants",
    "variant",
    "mutation",
    "mutants",
    "patients",
    "cells",
    "results",
    "mutant",
    "mutations",
] + list(stop_words)
