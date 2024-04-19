from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel

class Model():

    def __init__(self,
                 model_name: str,
    ):
        self.model_name = model_name
        

    @property
    def model(self):
        if "pythia" in self.model_name:
            return self.get_pythia_model(self.model_name.replace("pythia-", ""))[0]
        if "gpt2" in self.model_name:
            return self.get_gpt2_model(self.model_name.replace("gpt2-", ""))[0]
        if "Mistral" in self.model_name:
            return self.get_mistral_model(self.model_name.replace("Mistral-", ""))[0]
        if "mamba" in self.model_name:
            return self.get_mamba_model(self.model_name.replace("mamba-", ""))[0]
        else:
            raise ValueError(f"Unsupported model: {self.model_name}.")
    
    @property
    def tokenizer(self):
        if "pythia" in self.model_name:
            return self.get_pythia_model(self.model_name.replace("pythia-", ""))[1]
        if "gpt2" in self.model_name:
            return self.get_gpt2_model(self.model_name.replace("gpt2-", ""))[1]
        if "Mistral" in self.model_name:
            return self.get_mistral_model(self.model_name.replace("Mistral-", ""))[1]
        if "mamba" in self.model_name:
            return self.get_mamba_model(self.model_name.replace("mamba-", ""))[1]
        else:
            raise ValueError(f"Unsupported model: {self.model_name}.")

    @property
    def n_layers(self):
        return self.model.config.num_hidden_layers


    def get_model(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def get_pythia_model(self, size: str):
        assert size in ["14m", "31m", "70m", "160m", "410m", "1b", "2.8b", "6.9b", "12b"]
        self.full_model_name = f"EleutherAI/pythia-{size}"
        return self.get_model(self.full_model_name)

    def get_gpt2_model(self, size: str):
        assert size in ["gpt2", "small", "medium", "large", "xl"]
        if (size == "small" or size=="gpt2"):
            self.full_model_name = "gpt2"
        else:
            self.full_model_name = f"gpt2-{size}"
        return self.get_model(self.full_model_name)

    def get_bert_model(self, model_name: str):
        model = BertModel.from_pretrained(model_name)
        return model

    def get_mistral_model(self, size: str):
        assert size in ["7", "7x8"]
        self.full_model_name = f"mistralai/Mistral-{size}B-v0.1"
        return self.get_model(self.full_model_name)

    def get_mamba_model(self, size: str):
        assert size in ["130m", "370m", "790m", "1.4b", "2.8b"]
        self.full_model_name = f"state-spaces/mamba-{size}"
        return self.get_model(self.full_model_name)



def get_layers_to_enumerate(model) -> list:
        full_model_name = model.config._name_or_path
        if 'gpt' in full_model_name:
            return model.transformer.h
        elif 'pythia' in full_model_name:
            return model.gpt_neox.layers
        elif 'bert' in full_model_name:
            return smodel.encoder.layer
        elif 'Mistral' in full_model_name:
            return smodel.model.layers
        else:
            raise ValueError(f"Unsupported model: {full_model_name}.")