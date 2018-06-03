import re


def myPreprocessor(doc):
    return doc.replace(',', ' ').replace('\'', ' ').replace('*', ' ').replace('. ', ' ').replace('“', ' ').replace('”', ' ').replace('%', ' ').replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').replace('\n', ' ')


def myTokenizer(doc):
    tokens = re.findall(r'(?=\d*[A-z].*)\w\S+', doc)
    return tokens


def newTokenizer(doc):
    tokens = re.findall(r'(?=[^A-z]*[A-z][^A-z]*)(?=[\D]{3,}\s)(?=\b)\S+', doc)
    return tokens

# at least one letter, might contain hyphens, preceeded by white space. Second paranthesis makes sure we have a least three digits and/or letters
def tokenizeratleastthree(doc):
    tokens = re.findall(r'(?=[^A-z]*[A-z][^A-z]*)(?=[A-z0-9]+-?[A-z0-9]+-?[A-z0-9]+\s)(?<=\s)\w\S+\w', doc)
    return tokens

def tokenizernonumbers(doc):
    tokens = re.findall(r'(?=[^A-z]*[A-z][^A-z]*)(?=[^\d\s]{3,})(?<=\s)\w\S+\w', doc)
    return tokens


def tokenizeronlyletters(doc):
    tokens = re.findall(r'(?=[A-z]+-?[A-z]+-?[A-z]+\s)(?<=\s)\S{3,}', doc)
    return tokens

#takes at words with at least five letters
def tokenizerfiveletters(doc):
    tokens = re.findall(r'(?=[A-z]+-?[A-z]+-?[A-z]+\s)\S{5,}', doc)
    return tokens

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw
stopWords = ["fig", "figure", "et", "al", "table",  
        "data", "analysis", "analyze", "study",  
        "method", "result", "conclusion", "author",  
        "find", "found", "show", "perform",  
        "demonstrate", "evaluate", "discuss", "google", "scholar",   
        "pubmed",  "web", "science", "crossref", "supplementary", '(fig.)', '(figure', 'fig.', 'al.', 'did', 'thus,', '...', 'interestingly,', 'and/or', 'author'] + list(esw)

stopWordsExtended = ["variants", "variant", "mutation", "mutants", "patients", "cells", "results", "mutant", "mutations"] + list(stopWords)