from transformers import GPTNeoXForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from transformer_lens import utils, HookedTransformer, ActivationCache, patching
import torch
from torch import Tensor
from jaxtyping import Int, Float
from typing import List, Optional, Tuple


def patchscope(opt, device):

    torch.set_grad_enabled(False) # To save GPU memory because we only do inference

    source_model = get_model(opt.source_model)
    target_model = get_model(opt.target_model)

    source_prompt = "Amazon's former CEO attended Oscars"
    target_prompt = "cat->cat; 135->135; hello->hello: ?"
    #source_tokens = model.to_tokens(source_prompt, prepend_bos=True)
    #source_tokens = source_tokens.to(device)

    _, source_cache = source_model.run_with_cache(source_prompt)
    print(source_cache)

    position = 0
    layer = 2
    
    patch_residual_stream(target_model, position, layer, target_prompt, source_cache)

    return 



def get_model(model_name: str):
    """
    Loads source or target model.
    """

    if model_name == 'gpt2_small':
        model = HookedTransformer.from_pretrained("gpt2-small")

    return model


def patch_residual_stream(
    target_model: HookedTransformer,
    position: int,
    layer: int,
    target_prompt: str,
    source_cache: Float[Tensor, 'd_model']
):
    """
    Patches a residual stream vector into the target model.
    """

    def hook_fn(activations: Float[Tensor, "batch seq_len"],
                hook: HookPoint
    ) -> TT["batch", "seq_len"]:

        # modify activations (can be inplace)
        return activations

    target_model.run_with_hooks(
        target_tokens,
        return_type=None,
        fwd_hooks=[
            (lambda name: name.endswith("resid_pre"), hook_fn)
        ]
    )





