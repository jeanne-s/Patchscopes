import argparse


class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--source_model', 
                                '-M', 
                                type=str, 
                                default='gpt2_small',
                                choices=['gpt2_small', 'pythia-6.9b'], 
                                help='Source model architecture.')

        self.parser.add_argument('--target_model', 
                                '-M*', 
                                type=str, 
                                default='gpt2_small',
                                choices=['gpt2_small', 'pythia-6.9b'], 
                                help='Target model architecture.')
 
        
        return self.parser.parse_args()