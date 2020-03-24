import torch
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

from skimage import io, transform
from skimage.util import img_as_float


def get_train_valid_loader(data_dir,
                           batch_size,
                           image_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=True,
                           num_workers=4):
    """
    Utility function for loading and returning train and valid
    multi-process iterators. A sample 9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
#             normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomApply(
                [transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90)]
            ),
            transforms.ToTensor(),
#             normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
#             normalize,
        ])

    # load the dataset
    train_dataset = datasets.ImageFolder(
        root = data_dir,
        transform = train_transform,
    )

    valid_dataset = datasets.ImageFolder(
        root = data_dir,
        transform = valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers,
    )

    # visualize some normalized images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
#         X = images.numpy().transpose([0, 2, 3, 1])
#         plot_images(X, labels)
        npimg = make_grid(images, nrow=3).numpy()
        plt.figure(figsize = (2*3,2*3))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        print(' '.join('%5s' % labels[j].numpy() for j in range(8)))

    return (train_loader, valid_loader)


class TestDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.all_imgs = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.all_imgs[idx])
        image = io.imread(img_path)
        image = img_as_float(image)
        
        name = self.all_imgs[idx]
        name = re.sub("[^A-Z0-9]", "", str(name)).replace("JPG", "").replace("PNG", "").replace("JPEG", "").replace("JFIF", "")
        name = name[-6:]
        
        sample = {'image': image, 'name': name}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    

class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, name = sample['image'], sample['name']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        name = name

        return {'image': img, 'name': name}
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, name = sample['image'], sample['name']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'name': name}
    
    
    
def get_test_loader(data_dir,
                    batch_size,
                    image_size,
                    shuffle=True,
                    show_sample=True,
                    num_workers=4):
    
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    
    transform = transforms.Compose([Rescale(image_size),ToTensor(),normalize])
    
    test_dataset = TestDataset(root_dir=data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    
        # visualize some normalized images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=9,
        num_workers=num_workers,
        shuffle=shuffle,
        )
        
        data_iter = iter(sample_loader)
        images = data_iter.next()['image']
        npimg = make_grid(images, nrow=3).numpy()
        plt.figure(figsize = (2*3,2*3))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    return test_loader

