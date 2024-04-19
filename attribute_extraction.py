import torch
from models import Model, get_layers_to_enumerate
from typing import Union


class AttributeExtraction():
    
    def __init__(self,
                 source_model_name: str,
                 target_model_name: str,
                 input_prompt: str,
                 target_prompt: str,
                 source_str_token: str,
                 source_layers_ids: list[int],
                 target_layers_ids: list[int]
    ):
        self.source_model_name = source_model_name
        self.target_model_name = target_model_name
        self.input_prompt = input_prompt
        self.target_prompt = target_prompt
        self.source_str_token = source_str_token
        self.source_layers_ids = source_layers_ids
        self.target_layers_ids = target_layers_ids

        self.source_model_class = Model(self.source_model_name)
        self.source_model = self.source_model_class.model
        self.source_tokenizer = self.source_model_class.tokenizer
        self.target_model_class = Model(self.target_model_name)
        self.target_model = self.target_model_class.model
        self.target_tokenizer = self.target_model_class.tokenizer

        self.source_token_position = self.get_source_token_position()
        self.target_token_position = -1
        self.target_ids = self.target_tokenizer(self.target_prompt, return_tensors='pt', truncation=True)
    


    def get_source_token_position(self) -> int:
        """ Get the position of the source token in the input prompt.
        If the source token is present multiple times in the input prompt, we consider the position of its last occurrence.
        """
        tokenized_input_prompt: list[int] = self.source_tokenizer(self.input_prompt, return_tensors='pt', truncation=True)['input_ids'][0].tolist()
        source_token: Union[int, list[int]] = self.source_tokenizer(self.source_str_token, return_tensors='pt', truncation=True)['input_ids'][0].tolist()

        # If the source_str_token is composed of multiple tokens, we consider the position of its last token
        if isinstance(source_token, list):
            source_token = source_token[-1]

        indices = [i for i in range(len(tokenized_input_prompt)) if tokenized_input_prompt[i] == source_token]

        return indices[-1] #tokenized_input_prompt.index(source_token)


    def get_source_model_activations_on_input_prompt(self) -> torch.Tensor:
        """ Get the residual stream activations of the source model for the input prompt.
        
        Returns:
        source_model_activations_on_input_prompt: torch.Tensor [batch_size, seq_len, n_layers, hidden_size]
        """
        input_ids = self.source_tokenizer(self.input_prompt, return_tensors='pt', truncation=True)
        source_model_activations_on_input_prompt = torch.zeros((input_ids['input_ids'].shape[0], input_ids['input_ids'].shape[1], self.source_model_class.n_layers, self.source_model.config.hidden_size))

        def store_source_activations(layer_id):
            def hook_fn(module, input, output):
                source_model_activations_on_input_prompt[:, :, layer_id, :] = output[0].detach()
            return hook_fn
        
        hooks = []
        layers = get_layers_to_enumerate(self.source_model)

        for layer_id, layer in enumerate(layers):
            hook_handle = layer.register_forward_hook(
                store_source_activations(layer_id)
            )
            hooks.append(hook_handle)

        with torch.no_grad():
            _ = self.source_model(**input_ids)
        for h in hooks:
            h.remove()

        return source_model_activations_on_input_prompt


    def patch_target_model_one_layer(self,
                                     source_layer_id: int,
                                     target_layer_id: int,
                                     source_model_activations_on_input_prompt: torch.Tensor
    ) -> torch.Tensor:
        """ Patch the target model at the target layer with source model activations and returns the output logits.

        Returns:
        logits: torch.Tensor [batch_size, seq_len, n_vocab]
        """

        def patching_handler(source_layer_id: int,
                             target_token_position: int,
                             source_token_position: int):
            def patching_hook(module, input, output):
                output[0][:, target_token_position, :] = source_model_activations_on_input_prompt[:, source_token_position, source_layer_id, :]
            return patching_hook
            
        hook_handle = get_layers_to_enumerate(self.target_model)[target_layer_id].register_forward_hook(
            patching_handler(source_layer_id=source_layer_id, 
                             target_token_position=self.target_token_position,
                             source_token_position=self.source_token_position)
        ) 

        try:
            with torch.no_grad():
                outputs = self.target_model(**self.target_ids)
                logits = outputs.logits

        finally:
            hook_handle.remove()

        return logits  


    def patch_target_model_multiple_layers(self            
    ) -> torch.Tensor:
        """ Patch the target model at the target layers with source model activations and returns the output logits.

        Returns:
        logits_all_layers: torch.Tensor [len(self.source_layers_ids), len(self.target_layers_ids), batch_size, seq_len, n_vocab]
        """
        source_model_activations_on_input_prompt = self.get_source_model_activations_on_input_prompt()
        input_ids: torch.Tensor = self.target_tokenizer(self.target_prompt, return_tensors='pt', truncation=True)
        logits_all_layers = torch.zeros((len(self.source_layers_ids), len(self.target_layers_ids), input_ids['input_ids'].shape[0], input_ids['input_ids'].shape[1], self.target_model.config.vocab_size))

        for s, source_layer_id in enumerate(self.source_layers_ids):
            for t, target_layer_id in enumerate(self.target_layers_ids):
                logits: torch.Tensor = self.patch_target_model_one_layer(source_layer_id=source_layer_id, 
                                                                         target_layer_id=target_layer_id,
                                                                         source_model_activations_on_input_prompt=source_model_activations_on_input_prompt)
                logits_all_layers[s, t, :, :, :] = logits

        return logits_all_layers
