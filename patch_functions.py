from transformers import GPTNeoXForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from transformer_lens import utils, HookedTransformer, ActivationCache, patching, evals
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
import torch
from torch import Tensor
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import seaborn as sns
from data import *


def patchscope(opt, 
              device,
              ):

    torch.set_grad_enabled(False) # To save GPU memory because we only do inference

    source_model = get_model(opt.source_model, device)
    target_model = get_model(opt.target_model, device)
    #model_sanity_check(target_model)

    source_prompt = "Amazon's former CEO attended Oscars" #"L'homme a planté un arbre sur la grève"
    target_prompt = "cat->cat; 135->135; hello->hello; black->black; shoe->shoe; start->start; mean->mean; ?" #"chat->chat; 135->135; bonjour->bonjour; ?"
    print('Source prompt:', source_prompt, source_model.to_str_tokens(source_prompt))
    print('Source token:', source_model.to_str_tokens(source_prompt)[4])
    source_position = 4

    print('Target prompt:', target_prompt, target_model.to_str_tokens(target_prompt))
    print('Target token:', target_model.to_str_tokens(target_prompt)[-1])
    target_position = len(target_model.to_str_tokens(target_prompt))-1

    source_layer = 0
    target_layer = 0
    print(f'Source layer: {source_layer}, Target layer: {target_layer}')

    _, source_cache = source_model.run_with_cache(source_prompt)
    
    patch_activations(
        target_model, 
        source_position, 
        source_layer, 
        target_position,
        target_layer,
        target_prompt, 
        source_cache
    )

    eval_pile_dataset(target_model)

    return 


def extraction_of_specific_attributes(opt, device):
    """
    Performs the experiment described in paragraph '4.2 Extraction of Specific Attributes'.
    Generates and saves the corresponding figure (see Figure 3 from the paper) in 
    ```figures/extraction/{task}```.
    
    Input prompt: '... {subject} ... {object}'
    Output prompt: 
    """
    print(f'----- Extraction of specific attributes (task: {opt.task}) ----- ')

    source_model = get_model(opt.source_model, device)
    target_model = get_model(opt.target_model, device)

    input_prompts = pd.read_csv(f'data/input_prompts_{opt.task}.csv')

    # Get target prompt T ("The largest city in x"). Target position is the one of token 'x'
    target_prompt = get_target_prompt_from_task(opt.task)
    target_tokens = target_model.to_tokens(target_prompt)
    target_position = len(target_tokens) - 1 

    accuracy_df = pd.DataFrame(columns=['subject', 'source_layer', 'accuracy'])
    source_layers = np.arange(0, source_model.cfg.n_layers, dtype=int)
    target_layers = np.arange(0, target_model.cfg.n_layers, dtype=int)

    for index, row in input_prompts.iterrows(): # for each input prompt
        subject = row['subject']
        S = row['S']
        #print('Subject', subject, '---', source_model.to_str_tokens(S))

        # Get source position
        tokenized_subject = source_model.to_str_tokens(source_model.to_tokens(subject))[-1]
        source_position = source_model.get_token_position(tokenized_subject, S)
        print('source_position', source_position)
        _, source_cache = source_model.run_with_cache(S)

        for source_layer in source_layers: # for each source layer
            #print('source_layer', l)
        
            for target_layer in target_layers: # for each target layer
                #print('target_layer', l_star)

                predicted_tokens = patch_activations(
                                        target_model, 
                                        source_position, 
                                        source_layer, 
                                        target_position,
                                        target_layer,
                                        target_prompt, 
                                        source_cache
                )
                break
                
                #faire les 20 next tokens
                #if obj in 20 next tokens:
                    # accuracy_df = pd.concatenate(accuracy_df,
                    #                             pd.DataFrame({'subject': subject,
                    #                                          'source_layer': l,
                    #                                          'accuracy': 1}))
                    # break
                #elif l_star==len(target_layer)-1:
                    # accuracy_df = pd.concatenate(accuracy_df,
                    #                             pd.DataFrame({'subject': subject,
                    #                                          'source_layer': l,
                    #                                          'accuracy': 0})

        
        # plot
        # save plot

    return


def get_model(model_name: str = 'gpt2-small', device=None) -> HookedTransformer:
    """
    Loads source or target model.

    model_name: ['gpt2-small', 'pythia-6.9b']
    """
    return HookedTransformer.from_pretrained(model_name, device=device)


def patch_activations(
    target_model: HookedTransformer,
    source_position: int,
    source_layer: int,
    target_position: int,
    target_layer: int,
    target_prompt: str,
    source_cache: ActivationCache,
    activation_type: str = 'resid_pre',
    max_new_tokens: int = 20
):
    """
    Patches an activation vector into the target model.
    """

    source_cache = source_cache[activation_type, source_layer]
    predicted_tokens = []

    def hook_fn(target_activations: Float[Tensor, '...'],
                hook: HookPoint
    ) -> Float[Tensor, '...']:
        target_activations[:,target_position,:] = source_cache[:,source_position,:]
        return target_activations

    #target_prompt = target_model.generate(target_prompt, max_new_tokens=20)

    for i in range (0, max_new_tokens):
        print('i', i)   
        target_logits = target_model.run_with_hooks(
            target_prompt,
            return_type="logits",
            fwd_hooks=[
                (get_act_name(activation_type, target_layer), hook_fn)
            ]
        )
        prediction = target_logits.argmax(dim=-1).squeeze()[:-1]
        predicted_tokens.append(prediction)
        target_prompt = target_prompt + target_model.to_string(prediction)
        print(target_prompt)


    #print('Target model output :', target_model.to_string(prediction))
    print(target_model.to_str_tokens(target_model.to_string(target_prompt)))
    return predicted_tokens


def model_sanity_check(model: HookedTransformer):
    loss = evals.sanity_check(model)
    if loss < 5:
        print(f'{model.cfg.model_name} is probably OK.')
    else:
        print(f'{model.cfg.model_name} has a high loss. Maybe something went wrong.')
    return


def eval_pile_dataset(model: HookedTransformer):
    pile_data_loader = evals.make_pile_data_loader(model.tokenizer, batch_size=8)
    eval_result = evals.evaluate_on_dataset(model, pile_data_loader)
    print("Eval on Pile dataset:", eval_result)
    return


def get_target_prompt_from_task(task='country_currency'):
    dataset_path = os.path.join('relations/data/', get_task_type(task)+'/')
    
    with open(dataset_path + task + '.json', 'r') as f:
        relations_dict = json.load(f)
        target_prompt = relations_dict['prompt_templates'][0]

    return target_prompt.split("{}")[0] + 'x'