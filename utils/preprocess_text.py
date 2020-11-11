# -*- coding: utf-8 -*-
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw


def my_preprocessor(doc):
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


def my_tokenizer(doc):
    tokens = re.findall(r"(?=\d*[A-z].*)\w\S+", doc)
    return tokens


def new_tokenizer(doc):
    tokens = re.findall(r"(?=[^A-z]*[A-z][^A-z]*)(?=[\D]{3,}\s)(?=\b)\S+", doc)
    return tokens


# at least one letter, might contain hyphens, preceeded by white space. Second paranthesis makes sure we have a least three digits and/or letters
def tokenizer_at_least_three(doc):
    tokens = re.findall(
        r"(?=[^A-z]*[A-z][^A-z]*)(?=[A-z0-9]+-?[A-z0-9]+-?[A-z0-9]+\s)(?<=\s)\w\S+\w",
        doc,
    )
    return tokens


def tokenizernonumbers(doc):
    tokens = re.findall(r"(?=[^A-z]*[A-z][^A-z]*)(?=[^\d\s]{3,})(?<=\s)\w\S+\w", doc)
    return tokens


def tokenizeronlyletters(doc):
    tokens = re.findall(r"(?=[A-z]+-?[A-z]+-?[A-z]+\s)(?<=\s)\S{3,}", doc)
    return tokens


# takes at words with at least five letters
def tokenizerfiveletters(doc):
    tokens = re.findall(r"(?=[A-z]+-?[A-z]+-?[A-z]+\s)\S{5,}", doc)
    return tokens


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
