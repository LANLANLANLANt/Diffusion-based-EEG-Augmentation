from torch.utils import data


class EpochDataset(data.Dataset):
    def __init__(self, epoches, targets: Optional[np.array], mode='train'):
        self.epoches = epoches
        self.targets = targets
        self.mode = mode

    def __len__(self):
        return len(self.epoches)

    def __getitem__(self, idx):
        epoch = self.epoches[idx]
        target = self.targets[idx]

        return epoch, target


