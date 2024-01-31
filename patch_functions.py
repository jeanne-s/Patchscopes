from transformers import GPTNeoXForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from transformer_lens import utils, HookedTransformer, ActivationCache, patching, evals
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
import torch
from torch import Tensor
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import evaluate


def patchscope(opt, device):

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
    print('Target model output :', target_model.to_string(prediction))
    print(target_model.to_str_tokens(target_model.to_string(prediction)))
    return


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