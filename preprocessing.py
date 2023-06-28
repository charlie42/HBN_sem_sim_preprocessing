import pandas as pd
import numpy as np
import csv
import string
import nltk
from sentence_transformers import SentenceTransformer, util
import re


import explore_functions

item_col = "questions"

def read_data():
    df = pd.read_csv('data/item-names-cleaned.csv')
    return df

def remove_assessments(df):
    # Remove PreInt... assessments (PreInt in datadic col)

    assessments_to_remove = ["PreInt", "DailyMed", "Diagnosis_KSADS", "WHODAS_P", "WHODAS_SR", "Peds_Migraine", "APQ", "ESPAD", "ConsensusD",
                              'MRI_Track', 'PBQ', 'Pegboard', 'Physical', 'PPVT', 'RemoteTask']

    for assessment in assessments_to_remove:
        print('Removing', assessment, '...')
        df = df[~df['datadic'].str.contains(assessment, na=False)]
        print(df.shape[0], 'rows remain')

    return df

def remove_domains(df):

    domains_to_remove = ["Physical_Fitness_and_Status", "Vision", "Neurologic_Function", "Medical_Status_Measures"]

    for domain in domains_to_remove:
        print('Removing', domain, '...')
        df = df[~df['domains'].str.contains(domain, na=False)]
        print(df.shape[0], 'rows remain')

    return df

def drop_duplicates(df):
    print('Dropping duplicates...')
    df = df.drop_duplicates()
    print(df.shape[0], 'rows remain')
    return df

def drop_na(df):
    print('Dropping rows with na...')
    df = df.dropna(subset=[item_col])
    print(df.shape[0], 'rows remain')
    return df

def make_lowercase(df):
    print('Making lowercase...')
    df[item_col] = df[item_col].str.lower()
    return df

def remove_numbers(df):
    print('Removing numbers...')

    df[item_col] = df[item_col].str.replace('\d', '', regex=True)

    return df

def remove_total_scores(df):
    print('Removing total scores...')
    
    # Remove items where "keys" columns contains _total (lower or upper case) or _Score
    df = df[~(df['keys'].str.contains('_total', regex=False) | df['keys'].str.contains('_Total', regex=False) | df['keys'].str.contains('_Score', regex=False))]
    print(df.shape[0], 'rows remain')

    return df

def remove_punctuation(df):
    print('Removing punctuation...')

    cleaned_items = []

    for item in df[item_col]:
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space -- to not merge words separated by /
        item_cleaned = item.translate(translator)

        # Replace multiple spaces with one space
        item_cleaned = ' '.join(item_cleaned.split())

        cleaned_items.append(item_cleaned)

    df[item_col] = cleaned_items

    print(df[item_col].head())

    return df

    
def remove_common_stop_words(df):

    print('Removing common stop words...')

    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))

    cleaned_items = []

    # removes stopwords 
    for item in df[item_col]:
        cleaned_item = ' '.join([word for word in item.split() if word not in stop_words])
        cleaned_items.append(cleaned_item)

    df[item_col] = cleaned_items

    return df

def remove_domain_specific_stop_words(df):

    print('Removing domain-specific stop words...')

    stop_words = ['score', 'tscore', 'rank', 'range', 'raw', 'percentile', 'scaled', 'index',  'total', 'standard', 'sum', 'scores', 'scre', 'corrected', 'uncorrrected', 
                        'age', 'item', 'count', 'rescored', 'uncorrected', 'breakoff', 'yes', 'subscale', 'disorders', 'average', 'domain', 'subdomain', 'descriptive', 
                        'composite', 'diagnosis', 'specifier', 'criterion', 'invalid', 'completed', 'icd', "valid", "administration",

                  'child', 'childs', 'i', 'you', 'she', 'her', 'he', 'his', 'him', 'they', 'them', 'their', 'we', 'our', 'us', 'me', 'my', 'mine', 'your', 'yours',

                  'problem', 'problems', 'trouble', 'difficulty', 'deficit', 'complaints', 'behavior', 'difficulties', 'unable', 'able', 

                  'frequency', 'frequently', 'easily', 'lot', 'past', 'often', 'day', 'certain', 'something', 'little', 'unusually', 'current', 'much', 'too', 'always', 
                        'ever', 'unusual', 'usually', 'many', 'well', 'significant', 'overly', 'strong', 'per', 'almost', 'never', 'less', 'expected',

                  'would', 'g', 'e', 'describe', 'etc', 'seems', 'try', 'get', 'gets', 'feel', 'feels', 'felt', 'complain', 'seem', 'please', 

                  "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                  ]

    cleaned_items = []

    # removes stopwords 
    for item in df[item_col]:
        cleaned_item = ' '.join([word for word in item.split() if word not in stop_words])
        cleaned_items.append(cleaned_item)

    df[item_col] = cleaned_items

    return df

def remove_paraphrased(df):

    print("Removing very similar items...")
    
    # https://www.sbert.net/examples/applications/paraphrase-mining/README.html
    items = list(df[item_col])

    model = SentenceTransformer('all-MiniLM-L6-v2')

    paraphrases = util.paraphrase_mining(model, items, show_progress_bar=True)

    for paraphrase in paraphrases[0:10]:
        score, i, j = paraphrase
        print(f"sentences are equal: {items[i] == items[j]}")
        print(f"{items[i]} \t\t {items[j]} \t\t Score: {round(score, 4)}")
        
    df_paraphrases = pd.DataFrame(paraphrases, columns=['score', 'idx1', 'idx2'])
    print(df_paraphrases)

    df = df.reset_index()

    # Print all rows where score > 0.95
    for _, row in df_paraphrases.iterrows():
        if (row["score"] > 0.90) & (row["score"] < 0.93):
            idx1 = row['idx1'].astype(int)
            print(idx1, type(idx1))
            idx2 = row['idx2'].astype(int)
            print(df.iloc[idx1][item_col])
            print(df.iloc[idx2][item_col])
            print(row["score"])

    # Remove idx2 items if score >0.95
    idx2_to_remove = df_paraphrases[df_paraphrases['score'] > 0.92]['idx2'].tolist()

    df = df[~df.index.isin(idx2_to_remove)]

    print(df.shape[0], 'rows remain')

    return df

def remove_assessment_name_from_item(df):
    print('Removing assessment name from item...')

    print("1",df)
    for i, _ in df.iterrows():
        print(df.at[i,'datadic'])
        datadic = str(df.at[i,'datadic']).lower()
        df.at[i,item_col] = df.at[i, item_col].replace(datadic, '')
    print(df)

    return df

def write_to_csv(df):
    print('Writing to csv...')
    df.to_csv('data/item-names-cleaned-preprocessed.csv', index=False)
        

df = read_data()
print(df.shape)

df = remove_assessments(df)

df = remove_domains(df)

df = drop_duplicates(df)

df = drop_na(df)

df = make_lowercase(df)

df = remove_numbers(df)

df = remove_total_scores(df)

df = remove_common_stop_words(df)

df = remove_punctuation(df)

df = remove_common_stop_words(df)

df = remove_domain_specific_stop_words(df)

df = drop_duplicates(df)

df = remove_paraphrased(df)

print("0",df)
df = remove_assessment_name_from_item(df)


explore_functions.print_common_words(df[item_col], n=100)

write_to_csv(df)
