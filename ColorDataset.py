from torch.utils.data import Dataset
import os


class ColorDataset(Dataset):

    def __init__(self, root):
        self.root = root
        # basic and adv folders
        # download ALL
        pass

    def __getitem__(self, idx):
        # get graph
        pass

    def __len__(self):
        return len(self.data)


def prepare_folders(root, clear_up=False):
    if not os.path.exists(root):
        raise RuntimeError('root dataset path do not exist')
    if not os.path.exists(os.path.join(root, 'ColorDataset')):
        os.mkdir(os.path.join(root, 'ColorDataset'))
    if not os.path.exists(os.path.join(root, 'ColorDataset', 'basic')):
        os.mkdir(os.path.join(root, 'ColorDataset', 'basic'))
    else:
        if clear_up:
            [os.remove(os.path.join(root, 'ColorDataset', 'basic', f) for f in os.listdir(
                os.path.join(root, 'ColorDataset', 'basic')))]
    if not os.path.exists(os.path.join(root, 'ColorDataset', 'adv')):
        os.mkdir(os.path.join(root, 'ColorDataset', 'adv'))
    else:
        if clear_up:
            [os.remove(os.path.join(root, 'ColorDataset', 'adv', f) for f in os.listdir(
                os.path.join(root, 'ColorDataset', 'adv')))]