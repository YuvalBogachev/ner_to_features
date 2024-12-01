import pandas as pd
import spacy as sp
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def ner_to_feature(df: pd.DataFrame, text_field: str, ner_model: str, cutoffs: list[int], once_per_document: bool) -> dict[int, pd.DataFrame]:
    
    """
    Recieves as input a dataframe with a (long) text field and converts it to many features, by applying
    NER on the text field and applying a lambda function for every entity based on the cutoffs provided
    by the user.

    Args:
        df (DataFrame): a DataFrame.
        text_field (str): a text column of df.
        ner_model (str): the name of the NER model.
        cutoffs (list): a list of cutoffs for the counter of occurences.
        once_per_document (bool): whether to count each entity inside a document once or as many times as it occurs.
        
    Returns:
        dict: a mapping of cutoff to the matching NER generated features dataframe, or
        -1 if has an invalid argument.

    """
    
    # Input validation

    if not isinstance(df, pd.DataFrame):
        print("df isn't a dataframe!")
        return -1
    
    if text_field not in df.columns:
        print(f"{text_field} doesn't exist in DataFrame!")
        return -1

    # https://stackoverflow.com/questions/64047426/find-out-all-the-language-models-installed-in-spacy
    if ner_model in list(sp.info()['pipelines'].keys()):
        sp_m = sp.load(ner_model)
    else:
        print(f"Spacy model {ner_model} doesn't exist!")
        return -1
    
    if not all(isinstance(c, int) for c in cutoffs):
        print("Cutoffs should only be of ints!")
        return -1

    # Entity counter
    e_counter = defaultdict(int)

    # Count entities
    for text in df[text_field]:
        doc = sp_m(text)
        if once_per_document:
            for ent in set([ent.text for ent in doc.ents]):
                e_counter[ent] += 1
        else:
            for ent in [ent.text for ent in doc.ents]:
                e_counter[ent] += 1

    # Generate cutoff based datasets
    datasets = {}
    for cutoff in cutoffs:
        passers = [e for e in e_counter if e_counter[e] >= cutoff]
        df_l = df.copy()
        for passer in passers:
            df_l[f'f_entity_{passer}'] = df_l[text_field].apply(lambda s: 1 if passer in s else 0)
        datasets[cutoff] = df_l

    return datasets