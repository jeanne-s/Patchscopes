import argparse


class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--source_model', '-M', type=str, default='gpt2_small',
                                choices=['gpt2_small'], help='Source model architecture.')
        self.parser.add_argument('--target_model', '-M*', type=str, default='gpt2_small',
                                choices=['gpt2_small'], help='Target model architecture.')

        self.parser.add_argument('--source_layer', '-l', type=int, default=0,
                                help='Source model layer.')
        self.parser.add_argument('--source_position', '-i', type=int, default=0,
                                help='Position of the token to be explained.')
 
        

        return self.parser.parse_args()