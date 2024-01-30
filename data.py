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


def get_task_type(task='country_currency'):
    if task in ['country_currency', 'company_ceo', 'city_in_country']:
        return 'factual'
    elif task in ['fruit_inside_color', 'fruit outside_color', 'object_superclass']:
        return 'commonsense' 


def get_subobj_from_json(task='country_currency'):
    """
    Extract a list of subjects from json datasets provided by (Hernandez et al. 2023)
    """
    dataset_path = os.path.join('relations/data/', get_task_type(task)+'/')

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

    headers = {"Authorization": f"Bearer {config.api_key}"}
    def query():
        response = requests.get(API_URL, headers=headers)
        return response.json()

    for sub in ['United Kingdom']:#subjects_list:
        file_name = os.path.join('data', get_task_type(task), task, f'{sub.split()[0]}.json')
        if ' ' in sub:
            file_name = os.path.join('data', get_task_type(task), task, f'{sub.split()[0]}_{sub.split()[1]}.json')
            sub = f'%22'+ sub.split()[0] +'%20'+ sub.split()[1] +'%22'
        query_text = sub
        print(query_text)
        API_URL = f"https://datasets-server.huggingface.co/search?dataset=iohadrubin%2Fwikitext-103-raw-v1&config=default&split=train&query={query_text}&offset=0&length=30"
        data = query()

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    return 


def truncate_samples(task, model):
    task_type = get_task_type(task) 
    data_path = os.path.join('data', get_task_type(task), task)

    # [TODO]
    # pour tous les json
        # subject = title.replace('.json', '')
        # go through rows
        # trouve occurence de subject
        # select 20 tokens avant (donc il faut connaître le modèle)
        # vérifier que le model "correctly encodes the tuple"
        # save to a dataframe (subject, S)

    # return dataframe de prompt


subjects_list, objects_list = get_subobj_from_json()
get_S_from_wikitext(subjects_list, task='country_currency')
