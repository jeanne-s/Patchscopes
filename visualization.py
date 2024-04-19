import torch
import plotly
import plotly.express as px
import pandas as pd

from attribute_extraction import AttributeExtraction


class AttributeExtractionVizualisation():

    def __init__(self,
                 attribute_extraction_instance: AttributeExtraction = None,
                 logits: torch.Tensor = None
    ):
        self.attribute_extraction_instance = attribute_extraction_instance
        self.logits = logits


    def plot_logits_per_layer_heatmap(self) -> plotly.graph_objects:
        """ Plot the logits per layer heatmap.
        """

        target_layers: list[int] = self.attribute_extraction_instance.target_layers_ids
        source_layers: list[int] = self.attribute_extraction_instance.source_layers_ids
        n_source_layers = len(target_layers)
        n_target_layers = len(source_layers)

        if self.logits is None:
            self.logits: torch.Tensor = self.attribute_extraction_instance.patch_target_model_multiple_layers() # [len(source_layers_ids), len(target_layers_ids), batch_size, seq_len, n_vocab]
        
        probabilities: torch.Tensor = torch.nn.functional.softmax(self.logits, dim=-1)
        top_tokens: torch.Tensor = torch.argmax(probabilities, dim=-1) # [len(source_layers_ids), len(target_layers_ids), batch_size, seq_len]
        top_last_token: torch.Tensor = top_tokens[:, :, 0, -1] # [len(source_layers_ids), len(target_layers_ids)]


        token_proba_df = pd.DataFrame()
        for i, s in enumerate(source_layers):
            for j, t in enumerate(target_layers):
                temp_dict = {'source_layer_id': s, 
                             'target_layer_id': t,
                             'token_proba': probabilities[i, j, 0, -1, top_last_token[i, j]].item(),
                             'str_token': self.attribute_extraction_instance.target_tokenizer.decode(top_last_token[i, j])}
                token_proba_df = pd.concat([token_proba_df, pd.DataFrame(temp_dict, index=[0])], ignore_index=True)    
                token_proba_pivot = token_proba_df.pivot(index='source_layer_id', columns='target_layer_id', values='token_proba')
                str_token_pivot = token_proba_df.pivot(index='source_layer_id', columns='target_layer_id', values='str_token')

        fig = px.imshow(token_proba_pivot,
                        labels=dict(x="Target layer", y="Source layer", color="Token probability"),
                        x = [f'{i}' for i in target_layers],
                        y = [f'{i}' for i in source_layers],
                        aspect="auto",
                        color_continuous_scale='tealrose'
        )
        fig.update_traces(text = str_token_pivot, texttemplate="%{text}")

        self.print_input_output_prompt()

        return fig.show() 


    def print_input_output_prompt(self) -> None:
        
        str_prompt = self.attribute_extraction_instance.input_prompt
        source_tokenizer = self.attribute_extraction_instance.source_tokenizer
        tokens = source_tokenizer.encode(str_prompt)
        str_tokens = [source_tokenizer.decode(t) for t in tokens]

        print(f"Input prompt: {str_prompt}\nSource token: '{str_tokens[self.attribute_extraction_instance.source_token_position]}'")
        print(f'Output prompt: {self.attribute_extraction_instance.target_prompt}')