from transformers import GPTNeoXForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from transformer_lens import utils, HookedTransformer, ActivationCache, patching, evals
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
import torch
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from models import get_model_from_name
from data import *


def patchscopes_v1(opt, 
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

    return 


def extraction_of_specific_attributes(opt, device):
    """
    Performs the experiment described in paragraph '4.2 Extraction of Specific Attributes'.
    Generates and saves the corresponding figure (see Figure 3 from the paper) in 
    ```figures/extraction/{task}```.
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
        object = row['object']
        S = row['S']
        print('Subject:', subject, 'Object:', object)
        print('S', S[:30])

        # Get source position
        tokenized_subject = source_model.to_str_tokens(source_model.to_tokens(subject))[-1]
        source_position = source_model.get_token_position(tokenized_subject, S)
        #print('source_position', source_position)

        _, source_cache = source_model.run_with_cache(S)

        for source_layer in source_layers: 
        
            for target_layer in target_layers: 

                with torch.no_grad():
                    predicted_tokens, _ = patch_activations(
                                            target_model=target_model, 
                                            source_position=source_position, 
                                            source_layer=source_layer, 
                                            target_position=target_position,
                                            target_layer=target_layer,
                                            target_prompt=target_prompt, 
                                            source_cache=source_cache
                    )
                
                predicted_tokens = predicted_tokens.to(torch.int)
                if target_model.to_tokens(object)[-1][-1].item() in predicted_tokens:
                    print(target_model.to_str_tokens(predicted_tokens), 'acc 1')
                    accuracy_df = pd.concat([accuracy_df,
                                            pd.DataFrame({'subject': subject,
                                                         'source_layer': source_layer,
                                                         'accuracy': 1},
                                                         index=[0])
                    ])
                elif target_layer==len(target_layers)-1:
                    print(target_model.to_str_tokens(predicted_tokens), 'acc 0')
                    accuracy_df = pd.concat([accuracy_df,
                                            pd.DataFrame({'subject': subject,
                                                         'source_layer': source_layer,
                                                         'accuracy': 0},
                                                         index=[0])
                    ])


    fig = sns.relplot(
        data=accuracy_df, kind="line",
        x="source_layer", y="accuracy"
    )
    print(accuracy_df['subject'].value_counts())
    fig.savefig(f"figures/extraction_of_specific_attributes_{opt.task}.png")
    accuracy_df.to_csv(f'data/extraction_accuracies_{opt.task}.csv', index=False)

    return


def logitlens(opt, device, nb_positions=10):
    """
    Applies the logit lens (nostalgebraist 2020).
    """
    
    source_model = get_model(opt.source_model, device)
    target_model = get_model(opt.source_model, device)

    source_prompt = target_prompt = """Recent work has demonstrated substantial gains on many NLP tasks
    and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task.
    While typically task-agnostic in architecture, this method still requires task-specific fine-tuning
    datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a
    new language task from only a few examples or from simple instructions – something which current NLP
    systems still largely struggle to do. Here we show that scaling up language models greatly improves
    task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art
    finetuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion
    parameters,""".replace("\n    ", " ")
    #source_prompt = target_prompt = """I love plasma. I love plasma. I love plasma. I love plasma."""

    logits_df = pd.DataFrame(columns=['layer', 'position', 'logit', 'str_token'])

    _, source_cache = source_model.run_with_cache(source_prompt)
    source_tokens = source_model.to_tokens(source_prompt)
    nb_tokens = source_tokens.shape[-1]
    source_positions = np.arange(nb_tokens-nb_positions-1, nb_tokens-1, 1)
    source_layers = np.arange(0, source_model.cfg.n_layers, dtype=int)

    for source_position in tqdm(source_positions):

        for source_layer in source_layers:

            with torch.no_grad():
                predicted_tokens, target_logits = patch_activations(
                    target_model=target_model, 
                    source_position=source_position, 
                    source_layer=source_layer, 
                    target_position=source_position,
                    target_layer=source_model.cfg.n_layers-1,
                    target_prompt=target_prompt, 
                    source_cache=source_cache,
                    activation_type='resid_post'
                )
                predicted_tokens = predicted_tokens.to(torch.int)
        
                next_logit = torch.max(target_logits[0, source_position, :])

            logits_df = pd.concat([logits_df,
                                   pd.DataFrame({'layer': source_layer,
                                                 'position': source_position,
                                                 'logit': next_logit,
                                                 'str_token': target_model.to_str_tokens(predicted_tokens[source_position])},
                                                 index=[0])
                        ])
    
    df_wide = logits_df.pivot_table(index='layer', columns='position', values='logit')

    annot = []
    for x in range(0, df_wide.shape[0]):
        layer_list = []
        for y in range(0, nb_positions):
            str_token = logits_df[(logits_df['layer']==x)&(logits_df['position']==nb_tokens-nb_positions-1+y)]['str_token'].item()
            layer_list.append(str_token)
        annot.append(layer_list)
    
    x_axis_labels = [source_model.to_str_tokens(source_tokens[0, p]) for p in source_positions]

    fig, ax = plt.subplots(figsize=(16,12))
    fig = sns.heatmap(df_wide, annot=annot, fmt="", linewidths=.5, 
                     ax=ax, xticklabels=x_axis_labels)
    fig.invert_yaxis()
    plt.savefig("figures/logitlens_gpt-3.png")
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
    activation_type: str = 'resid_pre'
):
    """
    Patches an activation vector into the target model.
    """

    source_cache = source_cache[activation_type, source_layer]
    predicted_tokens = torch.Tensor()

    def hook_fn(target_activations: Float[Tensor, '...'],
                hook: HookPoint
    ) -> Float[Tensor, '...']:
        target_activations[:,target_position,:] = source_cache[:,source_position,:]
        return target_activations

    
    target_logits = target_model.run_with_hooks(
        target_prompt,
        return_type="logits",
        fwd_hooks=[
            (get_act_name(activation_type, target_layer), hook_fn)
        ]
    )
    prediction = target_logits.argmax(dim=-1).squeeze()[:-1]
    predicted_tokens = torch.cat((predicted_tokens, prediction))
    target_prompt = target_prompt + target_model.to_string(prediction)
    #print('target_prompt', target_prompt)

    return predicted_tokens, target_logits


def get_target_prompt_from_task(task='country_currency'):
    dataset_path = os.path.join('relations/data/', get_task_type(task)+'/')
    
    with open(dataset_path + task + '.json', 'r') as f:
        relations_dict = json.load(f)
        target_prompt = relations_dict['prompt_templates'][0]

    return target_prompt.split("{}")[0] #+ 'x'


def get_layers_to_enumerate(model):
    model_name = model.config._name_or_path
    if 'gpt' in model_name:
        return model.transformer.h
    elif 'pythia' in model_name:
        return model.gpt_neox.layers
    elif 'bert' in model_name:
        return model.encoder.layer
    elif 'Mistral' in model_name:
        return model.model.layers
    else:
        raise ValueError(f"Unsupported model: {model_name}.")


def get_residual_stream_activations(model,
                                    tokenizer,
                                    prompt: str,
                                    token_pos: int,
                                    n_samples: int = 1
):
    """ Returns a torch.Tensor of activations. 

    activations.shape = torch.Size([batch, num_hidden_layers, hidden_size])
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True)
    activations = torch.zeros((input_ids.shape[0], input_ids.shape[1], model.config.num_hidden_layers, model.config.hidden_size)) # batch, seq, n_layers, d_model

    def get_activation(layer_id):
        def hook(model, input, output):
            activations[:, :, layer_id, :] = output[0].detach()
        return hook

    layers_to_enum = get_layers_to_enumerate(model)
    hooks = []
    for i, layer in enumerate(layers_to_enum):
        hook_handle = layer.register_forward_hook(get_activation(i))
        hooks.append(hook_handle)

    _ = model(input_ids)
    for h in hooks:
        h.remove()
    return activations


def apply_activation_patch(model, 
                           tokenizer, 
                           target_prompt,
                           source_activations, #: Float[torch.Tensor, 'batch seq n_layers d_model'], 
                           target_layer_idx: int = 2, 
                           target_token_pos: int = -1, 
):

    def patch_activations(source_activations, 
                          target_layer_idx, 
                          target_token_idx):
        def hook(module, input, output):
            output[0][:, target_token_pos, :] = source_activations[:, target_token_pos, target_layer_idx, :]
        return hook

    input_ids = tokenizer(target_prompt, return_tensors="pt", truncation=True)
    
    layers = get_layers_to_enumerate(model)
    target_layer = layers[target_layer_idx]
    hook_handle = target_layer.register_forward_hook(
        patch_activations(source_activations, target_layer_idx, target_token_pos)
    )
    
    try:
        with torch.no_grad():
            tokens = model.generate(**input_ids)
    finally:
        hook_handle.remove()
    
    return tokens


def patchscopes(source_model_name: str,
                     target_model_name: str,
                     source_prompt: str,
                     target_prompt: str,
                     source_token_pos: int,
                     target_token_pos: int
):

    source_model, source_tokenizer = get_model_from_name(source_model_name)

    if target_model_name == source_model_name:
        target_model, target_tokenizer = source_model, source_tokenizer
    else:
        target_model, target_tokenizer = get_model_from_name(target_model_name)

    target_layers = len(get_layers_to_enumerate(target_model))

    source_activations = get_residual_stream_activations(model=source_model,
                                                         tokenizer=source_tokenizer,
                                                         prompt=source_prompt,
                                                         token_pos=source_token_pos)


    for layer in range(target_layers):
        tokens = apply_activation_patch(model=target_model,
                                        tokenizer=target_tokenizer,
                                        target_prompt=target_prompt,
                                        target_layer_idx=layer,
                                        target_token_pos=target_token_pos,
                                        source_activations=source_activations)

        str_tokens = target_tokenizer.batch_decode(tokens)
        print(f'Layer {layer} - {str_tokens}')
    return

