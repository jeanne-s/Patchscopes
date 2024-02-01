import os
import json
from datasets import load_dataset
import pandas as pd 
import numpy as np
from tqdm import tqdm
import requests
import config
from transformer_lens import utils, HookedTransformer
from base_options import *


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

    relation = relations_dict['properties']['range_name']

    return subjects_list, objects_list, relation


def get_S_from_wikitext(subjects_list, objects_list, task):

    data_path = os.path.join('data', get_task_type(task), task)
    samples_df = pd.DataFrame(columns=['subject', 'object', 'S'])

    headers = {"Authorization": f"Bearer {config.api_key}"}
    def query():
        response = requests.get(API_URL, headers=headers)
        return response.json()

    for idx, sub in enumerate(tqdm(subjects_list)):
        print(sub)
        obj = objects_list[idx]
        fetch_n = 0

        while (samples_df['subject'].values == sub).sum() < 20 and fetch_n < 60:
            file_name = os.path.join('data', get_task_type(task), task, f'{sub.split()[0]}.json')
            if ' ' in sub:
                file_name = os.path.join('data', get_task_type(task), task, f'{sub.split()[0]}_{sub.split()[1]}.json')
                sub_form = '%22'+ sub.split()[0] +'%20'+ sub.split()[1] +'%22'
                query_text = sub_form 
            else:
                query_text = sub 
            API_URL = f"https://datasets-server.huggingface.co/search?dataset=iohadrubin%2Fwikitext-103-raw-v1&config=default&split=train&query={query_text}&offset=0&length=100"
            data = query() # dict{'rows':{'row': {'text': .....}}}

            for r in data['rows']:
                text = text = r['row']['text']
                obj_index = text.rfind(obj.lower())
                if obj_index == -1:
                    continue
                else:
                    subj_index = text.rfind(sub, 0, obj_index)
                    if subj_index != -1:
                        if text[subj_index:subj_index+len(sub)+1]==sub+' ' and text[obj_index-1:obj_index+len(obj)]==' '+obj.lower():
                            samples_df = pd.concat([samples_df, 
                                                    pd.DataFrame(
                                                        {'subject': sub,
                                                        'object': obj, 
                                                        'S': text[subj_index: obj_index+len(obj)]},
                                                        index=[0])
                            ])
                            samples_df.to_csv(f'data/input_prompts_{task}.csv', index=False)
            fetch_n += 1
            if fetch_n==60: print(f'{sub} not found') 

    print(f'Saved {data_path}/input_prompts_{task}.csv')
    print(samples_df['subject'].value_counts())                
    samples_df.to_csv(f'{data_path}/input_prompts_{task}.csv', index=False)
    return 


def clean_input_prompts(task='country_currency', model_name='gpt2-small'):
    input_df = pd.read_csv(f'data/input_prompts_{task}.csv')
    output_df = pd.DataFrame(columns=['subject', 'object', 'S'])
    model = HookedTransformer.from_pretrained(model_name, device=None)
    
    for index, row in input_df.iterrows():
        subj, obj, text = row['subject'], row['object'], row['S']
        tokens = model.to_str_tokens(text)
        if subj in tokens[-20:]:
            print(tokens[-20:])
            output_df = pd.concat([output_df, pd.DataFrame({'subject': subj,
                                                            'object': obj, 
                                                            'S': text},
                                                            index=[0])
            ])

    print(output_df['subject'].value_counts())
    output_df.to_csv(f'data/cleaned_input_prompts_{task}.csv', index=False)
    return
    



subjects_list, objects_list, relation = get_subobj_from_json()
get_S_from_wikitext(subjects_list, objects_list, task='country_currency')

clean_input_prompts()
