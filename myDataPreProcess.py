# -*- coding: utf-8 -*-
"""
Data Loading and Processing Tutorial
====================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

This is the original author of the original tutorial.

**New Author**: Yang Wang

A lot of effort in solving any machine learning problem goes in to
preparing the data. PyTorch provides many tools to make data loading
easy and hopefully, to make your code more readable. In this tutorial,
we will see how to load and preprocess/augment data from a non trivial
dataset.

To run this tutorial, please make sure the following packages are
installed:

-  ``scikit-image``: For image io and transforms
-  ``pandas``: For easier csv parsing

This is a modified version of the original code.

The job for this script is still pre-processing but with a different "dataset".
To use nice built-in functions and frameworks of PyTorch, 
I have to modify to code to interface with the overall environment.
The idea is to get some data that we want to learn a model out of it.
output_data = mapping(input_data), the mapping is our model, what we try to learn.
The (input_data, output_data) pair is our dataset.
And this script is to prepare for the next step, which is machine learning.

raw input of this script: 2 csv files for input_data and output_data, respectively

output of this script:  

a dict that contains a batch of data from the dataset

key1: jointspace, input_data, type: tensor, shape: [batchSize, someDim, 1, 1]

key2: workspace, output_data, type: tensor, shape: [batchSize, someDim, 1, 1]

someDim corresponds to the length of raw data of each sample

the following is trying doctest material but didn't work

>>>1+1
2

"""

from __future__ import print_function, division
import os
import sys
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


######################################################################
# Dataset class
# -------------
#
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset.
# Your custom dataset should inherit ``Dataset`` and override the following
# methods:
#
# -  ``__len__`` so that ``len(dataset)`` returns the size of the dataset.
# -  ``__getitem__`` to support the indexing such that ``dataset[i]`` can
#    be used to get :math:`i`\ th sample
#
# Let's create a dataset class for our Forward Kinematic dataset. We will
# read the whole csv in ``__init__`` and get one sample in
# ``__getitem__``. This could be memory efficient for other cases
# but not very obvious in our case.
#
# Sample of our dataset will be a dict
# ``{'jointspace': jointangles, 'workspace': endeffposes}``. Our datset will take an
# optional argument ``transform`` so that any required processing can be
# applied on the sample. We will see the usefulness of ``transform`` in the
# next section.
#

class ForKinDataset(Dataset):
    """

    Forward Kinematics dataset.

    One Dataset should contain multiple samples 
    and each sample should have the corresponding I and O.

    """

    def __init__(self, csv_JS, csv_WS, root_dir, transform=None):
        """
        Args:
            csv_JS (string): Path to the csv file with input.
            csv_WS (string): Path to the csv file with output/labels/annotations.
            root_dir (string): Directory with all the raw csv files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.jointangles_frame = pd.read_csv(csv_JS)
        self.endeffposes_frame = pd.read_csv(csv_WS)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """

        return nSamples in the Dataset

        use try/except to make sure I and O have the same number of entries

        ########################################################################
        # .. note::
        #
        #     due to ``print(f"strings")``, python version >=3.6

        """
        try:
            input_len = len(self.jointangles_frame)
            output_len = len(self.endeffposes_frame)
            assert input_len == output_len
        except AssertionError:
            print(f"Length not equal: len(input) == {input_len} but len(output) == {output_len}")
            # sys.exit(1)
            return 0
        return len(self.jointangles_frame)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): index for a specific sample which ideally could be the "name" of the sample

        """
        jointangles = self.jointangles_frame.iloc[idx, :].values
        # jointangles = jointangles.astype('float').reshape(-1, 2)
        # i dont think this recast and reshape is relevant with my case
        endeffposes = self.endeffposes_frame.iloc[idx, :].values
        sample = {'jointspace': jointangles, 'workspace': endeffposes}

        if self.transform:
            sample = self.transform(sample)

        return sample


# ######################################################################
# # Let's instantiate this class and iterate through the data samples. We
# # will print the sizes of first 4 samples and show their landmarks.
# #

# fk_dataset = ForKinDataset( csv_JS='workdir/saveJS.csv',
#                             csv_WS='workdir/saveWS.csv',
#                             root_dir='workdir/')

# # fig = plt.figure()

# for i in range(len(fk_dataset)):
#     sample = fk_dataset[i]

#     print(i, sample['jointspace'].shape, sample['workspace'].shape)
#     # print(type(sample['jointspace']))

#     if i == 3:
#         break


######################################################################
# Transforms
# ----------
#
# Our samples are of the same size, which is good, 
# so that we don't need rescaling or cropping.
# But still, we will need at least one to turn ndarray to tensor.
#
# -  ``ToTensor``: to convert the numpy images to torch images (we need to
#    swap axes).
#
# We will write them as callable classes instead of simple functions so
# that parameters of the transform need not be passed everytime it's
# called. For this, we just need to implement ``__call__`` method and
# if required, ``__init__`` method. We can then use a transform like this:
#
# ::
#
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)
#
# Of course, we could later add other transforms to augment the dataset,
# such as adding noise to the original samples.
# Whether or not that's a good idea for training is another problem to ask.
#
# Observe below how these transforms had to be applied both on the I and O.
#

# TODO: the two above need to go and maybe turn one into augmentation with noise
# TODO: maybe do a normalization like stated in the last part of comment

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.

    Again, this makes more sense if this is an image dataset
    for you have to swap color axis because of different conventions. ::

        numpy image: H x W x C
        torch image: C X H X W

    But for our dataset, it's more about having all the dimensions needed
    for the framework to work properly.

    ``torch.unsqueeze()`` is used to do that.

    """

    def __call__(self, sample):
        """
        `unsqueeze` some dimensions to match the shape
        
        """
        jointangles, endeffposes = sample['jointspace'], sample['workspace']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # jointangles = jointangles.transpose((2, 0, 1))
        # endeffposes = endeffposes.transpose((2, 0, 1))
        tensor_jointangles = torch.from_numpy(jointangles)
        tensor_endeffposes = torch.from_numpy(endeffposes)
        tensor_jointangles = torch.unsqueeze(tensor_jointangles,0)
        tensor_endeffposes = torch.unsqueeze(tensor_endeffposes,0)
        tensor_jointangles = torch.unsqueeze(tensor_jointangles,2)
        tensor_endeffposes = torch.unsqueeze(tensor_endeffposes,2)
        return {'jointspace': tensor_jointangles,
                'workspace': tensor_endeffposes}


######################################################################
# Compose transforms
# ~~~~~~~~~~~~~~~~~~
#
# Now, we apply the transforms on an sample.
#
# although we don't have multiple transforms at the moment,
# they are actually composable.
# ``torchvision.transforms.Compose`` is a simple callable class which allows us
# to do this.
#

# scale = Rescale(256)
# crop = RandomCrop(128)
# composed = transforms.Compose([Rescale(256),
#                                RandomCrop(224)])

# # Apply each of the above transforms on sample.
# fig = plt.figure()
# sample = face_dataset[65]
# for i, tsfrm in enumerate([scale, crop, composed]):
#     transformed_sample = tsfrm(sample)

#     ax = plt.subplot(1, 3, i + 1)
#     plt.tight_layout()
#     ax.set_title(type(tsfrm).__name__)
#     show_landmarks(**transformed_sample)

# plt.show()


# ######################################################################
# # Iterating through the dataset
# # -----------------------------
# #
# # Let's put this all together to create a dataset with composed
# # transforms.
# # To summarize, every time this dataset is sampled:
# #
# # -  An image is read from the file on the fly
# # -  Transforms are applied on the read image
# # -  Since one of the transforms is random, data is augmentated on
# #    sampling
# #
# # We can iterate over the created dataset with a ``for i in range``
# # loop as before.
# #

# transformed_dataset = ForKinDataset(csv_JS='workdir/saveJS.csv',
#                                     csv_WS='workdir/saveWS.csv',
#                                     root_dir='workdir/',
#                                     transform=transforms.Compose([
#                                                ToTensor()
#                                     ]))

# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]

#     print(i, sample['jointspace'].size(), sample['workspace'].size())

#     if i == 3:
#         break


# ######################################################################
# # However, we are losing a lot of features by using a simple ``for`` loop to
# # iterate over the data. In particular, we are missing out on:
# #
# # -  Batching the data
# # -  Shuffling the data
# # -  Load the data in parallel using ``multiprocessing`` workers.
# #
# # ``torch.utils.data.DataLoader`` is an iterator which provides all these
# # features. Parameters used below should be clear. One parameter of
# # interest is ``collate_fn``. You can specify how exactly the samples need
# # to be batched using ``collate_fn``. However, default collate should work
# # fine for most use cases.
# #

# dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=4)


# # # Helper function to show a batch
# # def show_landmarks_batch(sample_batched):
# #     """Show image with landmarks for a batch of samples."""
# #     images_batch, landmarks_batch = \
# #             sample_batched['image'], sample_batched['landmarks']
# #     batch_size = len(images_batch)
# #     im_size = images_batch.size(2)

# #     grid = utils.make_grid(images_batch)
# #     plt.imshow(grid.numpy().transpose((1, 2, 0)))

# #     for i in range(batch_size):
# #         plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
# #                     landmarks_batch[i, :, 1].numpy(),
# #                     s=10, marker='.', c='r')

# #         plt.title('Batch from dataloader')

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['jointspace'].size(),
#           sample_batched['workspace'].size())

#     # observe 4th batch and stop.
#     if i_batch == 3:
#         break

# import pdb
# pdb.set_trace()

# if __name__ == "__main__": 
#     import doctest
#     doctest.testmod()
    
# ######################################################################
# # Afterword: torchvision
# # ----------------------
# #
# # In this tutorial, we have seen how to write and use datasets, transforms
# # and dataloader. ``torchvision`` package provides some common datasets and
# # transforms. You might not even have to write custom classes. One of the
# # more generic datasets available in torchvision is ``ImageFolder``.
# # It assumes that images are organized in the following way: ::
# #
# #     root/ants/xxx.png
# #     root/ants/xxy.jpeg
# #     root/ants/xxz.png
# #     .
# #     .
# #     .
# #     root/bees/123.jpg
# #     root/bees/nsdf3.png
# #     root/bees/asd932_.png
# #
# # where 'ants', 'bees' etc. are class labels. Similarly generic transforms
# # which operate on ``PIL.Image`` like  ``RandomHorizontalFlip``, ``Scale``,
# # are also available. You can use these to write a dataloader like this: ::
# #
# #   import torch
# #   from torchvision import transforms, datasets
# #
# #   data_transform = transforms.Compose([
# #           transforms.RandomSizedCrop(224),
# #           transforms.RandomHorizontalFlip(),
# #           transforms.ToTensor(),
# #           transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                                std=[0.229, 0.224, 0.225])
# #       ])
# #   hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
# #                                              transform=data_transform)
# #   dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
# #                                                batch_size=4, shuffle=True,
# #                                                num_workers=4)
# #
# # For an example with training code, please see
# # :doc:`transfer_learning_tutorial`.
