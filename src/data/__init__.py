from .dataset import InpaintingData

from torch.utils.data import DataLoader


def sample_data(loader): 
    while True:
        for batch in loader:
            yield batch


def create_loader(args): 
    print("Inside data loader function.....")
    dataset = InpaintingData(args)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size//args.world_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    print("DATA loaded.....")
    
    return sample_data(data_loader)
