import pandas as pd

def load_data():
    essays = pd.read_csv("./data/essay-br.csv")
    prompts = pd.read_csv("./data/prompts.csv")
    return essays, prompts


def categorize_score(score):
    return score // 40