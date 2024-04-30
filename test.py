import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

if __name__ == '__main__':
    print("start testing")
    '''
    GPU usage
    '''
    if torch.cuda.is_available():
        print("==============> Using GPU")
        device = 'cuda:0'
    else:
        print("==============> Using CPU")
        device = 'cpu'

    '''
    Seed
    '''
    myseed = 7414  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    '''
    Model
    '''
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 26)
    )
    model.eval()
    '''
    Apply transformation
    '''
    input = np.load("hand.npy").to(device)
    output = model(input)
    print(output)



