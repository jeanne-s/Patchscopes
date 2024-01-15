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

    _, cache = source_model.run_with_cache(source_prompt)
    print(cache)




def get_model(model_name):
    """
    Loads source or target model.

    model_name: str ['pythia-70b-deduped']
    """

    if model_name == 'gpt2_small':
        model = HookedTransformer.from_pretrained("gpt2-small")

    else:
        raise exception_factory(ValueError, "Invalid model_name.")

    return model


def patch_residual_stream(
    target_model: HookedTransformer,
    position: int,
    layer: int,
    target_tokens: Float[Tensor, 'batch pos'],
    residual_stream_vector: Float[Tensor, 'd_model']
):
    """
    Patches a residual stream vector into the target model.
    """

    def hook_fn(residual_stream: Float[Tensor, "batch seq_len"],
                hook: HookPoint
    ) -> TT["batch", "seq_len"]:

        # modify residual_stream (can be inplace)
        return residual_stream

    target_model.run_with_hooks(
        target_tokens,
        return_type=None,
        fwd_hooks=[
            (lambda name: name.endswith("resid_pre"), hook_fn)
        ]
    )





