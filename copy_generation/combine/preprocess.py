from collections import defaultdict
from typing import List
import pandas as pd

def tokenize_column(df: pd.DataFrame, *, col_to_tokenize: str, tokenized_col_name: str, inplace=False) -> pd.DataFrame or None:
    """
    Tokenize text column and create new column with tokenized text
    """

    if inplace:
        df[tokenized_col_name] = df[col_to_tokenize].str.split(' ')
    else:
        df_copy = df.copy()
        df_copy[tokenized_col_name] = df_copy[col_to_tokenize].str.split(' ')
        return df_copy
    
    return None
    

def create_binary_seq_from_row(row: pd.Series, code_token_column: str, pseudo_token_column: str) -> list:
    """
    Returns binary sequence for pseudocode tokens based on true code tokens

    If the pseudocode token exists in the true code (i.e., can be copied), then 
    the sequence contains a 1 in that position. Otherwise, the sequence contains
    a 0.
    """
    code_token_set = set(row[code_token_column])
    output_seq = []

    for token in row[pseudo_token_column]:
        if token in code_token_set:
            output_seq.append(1)
        else:
            output_seq.append(0)

    assert len(output_seq) == len(row[pseudo_token_column])
    return output_seq


def create_tagged_tuples(row: pd.Series, pseudo_token_column: str, code_binary_seq_column: str) -> List[tuple]:
    """
    Zips pseudo_token and code_binary_seq
    """
    z = zip(row[pseudo_token_column], row[code_binary_seq_column])
    return list(z)


def create_features(row: pd.Series, pseudo_token_column: str) -> dict:
    """
    Creates features of each pseudocode token
    """

    features = []

    for index, word in enumerate(row[pseudo_token_column]):
        features.append({
            'word': word,
            'length': len(word),
            'is_numeric': word.isnumeric(),
            'is_alpha': word.isalpha(),
            'is_alphanumeric': word.isalnum(),
            'is_punctuation': word.isalpha(),
            'prev_word': '' if index == 0 else row[pseudo_token_column][index-1],
            'next_word': '' if index == len(row[pseudo_token_column])-1 else row[pseudo_token_column][index+1],
            'prev_prev_word': '' if index <= 1 else row[pseudo_token_column][index-2],
            'next_next_word': '' if index >= len(row[pseudo_token_column])-2 else row[pseudo_token_column][index+2],
        })

    return features

def create_indices(row: pd.Series, col_name: str, lexicon: defaultdict) -> pd.Series:
    """
    Create indices for each token in the column
    """
    indices = []
    for word in row[col_name]:
        indices.append(lexicon[word])
    return indices


def create_combine_features(row: pd.Series, cpy_token_column: str, pseudo_tokens_column: str, input_lexicon: dict) -> dict:
    """
    Creates features of each pseudocode token
    """

    features = []
    corresponding_pseudo_tokens = iter(row[pseudo_tokens_column])

    for index, word in enumerate(row[cpy_token_column]):
        features.append({
            'word': input_lexicon[word],
            'length': len(word),
            'is_numeric': word.isnumeric(),
            'is_alpha': word.isalpha(),
            'is_alphanumeric': word.isalnum(),
            'is_punctuation': word.isalpha(),
            'prev_word': -1 if index == 0 else input_lexicon[row[cpy_token_column][index-1]],
            'next_word': -1 if index == len(row[cpy_token_column])-1 else input_lexicon[row[cpy_token_column][index+1]],
            'prev_prev_word': -1 if index <= 1 else input_lexicon[row[cpy_token_column][index-2]],
            'next_next_word': -1 if index >= len(row[cpy_token_column])-2 else input_lexicon[row[cpy_token_column][index+2]],
            'is_cpy_token': True if word == '[CPY]' else False,
            'cpy_word_index': input_lexicon[next(corresponding_pseudo_tokens, '')],
        })

    return features


def apply_function_to_column(row: pd.Series, function: callable, col_name: str, *args, **kwargs) -> pd.Series:
    """
    Apply function to column of dataframe row
    """
    return function(row[col_name], *args, **kwargs)