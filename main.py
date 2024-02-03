from base_options import *
from patch_functions import *
import torch

if __name__ == '__main__':

    opt = BaseOptions().parse()
    print('\nPatchscopes options\n-------------------')
    print(''.join(f'{k} = {v}\n' for k, v in vars(opt).items()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device =', device, '\n-------------------')

    if opt.experiment=='extraction':
        extraction_of_specific_attributes(opt, device)
    elif opt.experiment=='logitlens':
        logitlens(opt, device)

    else:
        patchscope(opt, device)
