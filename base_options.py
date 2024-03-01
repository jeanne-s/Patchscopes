import argparse


class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--source_model', 
                                '-M', 
                                type=str, 
                                default='gpt2-small',
                                help='Source model architecture.')

        self.parser.add_argument('--target_model', 
                                '-M*', 
                                type=str, 
                                default='gpt2-small',
                                choices=['gpt2-small', 'pythia-6.9b', 'gpt-j-6B'], 
                                help='Target model architecture.')

        self.parser.add_argument('--experiment',
                                '-e',
                                type=str,
                                default='extraction',
                                choices=['extraction', 'logitlens'],
                                help="Type of experiment. 'extraction': paragraph 4.2. 'logitlens' (nostalgebraist 2020). ")

        self.parser.add_argument('--task',
                                '-t',
                                type=str,
                                default='country_currency',
                                help="Task label from the relations dataset by Hernandez et al. 2023. for the experiment '4.2 Extraction of Specific Attributes")
 
        
        return self.parser.parse_args()