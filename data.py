import os
import json
from datasets import load_dataset
import pandas as pd 
from tqdm import tqdm
import requests
import config


def get_wikitext_103(split='train', n_samples=1):
    dataset = load_dataset("iohadrubin/wikitext-103-raw-v1", split=split, streaming=True)
    data = list(dataset.take(n_samples)['text'])
    return data


def get_subobj_from_json(task='country_currency'):
    """
    Extract a list of subjects from json datasets provided by (Hernandez et al. 2023)
    """
    if task in ['country_currency']:
        dataset_path = os.path.join('relations/data/', 'factual/')
    elif task in ['fruit_inside_color']:
        dataset_path = os.path.join('relations/data/', 'commonsense/')

    with open(dataset_path + task + '.json', 'r') as f:
        relations_dict = json.load(f)

    subjects_list = []
    objects_list = []
    samples = relations_dict['samples']
    for e in samples:
        subjects_list.append(e['subject'])
        objects_list.append(e['object'])

    return subjects_list, objects_list


def get_S_from_wikitext(subjects_list, task):

    source_prompts = pd.DataFrame(columns=['subject', 'S'])

    dataset = load_dataset("iohadrubin/wikitext-103-raw-v1", split='train', streaming=True)

    headers = {"Authorization": f"Bearer {config.api_key}"}
    API_URL = "https://datasets-server.huggingface.co/rows?dataset=iohadrubin%2Fwikitext-103-raw-v1&config=default&split=train&offset=0&length=100"
    def query():
        response = requests.get(API_URL, headers=headers)
        return response.json()
    data = query()



    for sub in tqdm(subjects_list):
        n = 0
        while n<5:
            data = list(dataset.take(1))[0]['text']
            if sub in data:
                source_prompts.append({'subject': sub, 'S': data})
                n += 1

            dataset.to_csv(f'wikitext_{task}.csv', index=False)
    
    return source_prompts


subjects_list, objects_list = get_subobj_from_json()
source_prompts = get_S_from_wikitext(subjects_list, task='country_currency')
print(source_prompts.info)

