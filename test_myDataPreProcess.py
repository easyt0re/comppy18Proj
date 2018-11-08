"""
Test Script for `pytest` or Travis-CI
========================================================================
**Author**: Yang Wang

Coverall is a good tool to push for test developing.

"""

import pytest
import pandas as pd
import myDataPreProcess

import torchvision.transforms as transforms

# init dataset
test_dataset = myDataPreProcess.ForKinDataset(	csv_JS="workdir/saveJS.csv",
												csv_WS="workdir/saveWS.csv",
												root_dir="workdir/")
tran_dataset = myDataPreProcess.ForKinDataset(	csv_JS="workdir/saveJS.csv",
												csv_WS="workdir/saveWS.csv",
												root_dir="workdir/",
												transform=transforms.Compose([myDataPreProcess.ToTensor()]))
# for some reason, Dataset doesn't have copy()
# so the dumb way is to initialize another one
bad_dataset = myDataPreProcess.ForKinDataset(	csv_JS="workdir/saveJS.csv",
												csv_WS="workdir/saveWS.csv",
												root_dir="workdir/",
												transform=transforms.Compose([myDataPreProcess.ToTensor()]))
# use the drop to make unmatched I/O to trigger try except
# df.drop(df.index[[1,3]])
# df.drop(df.tail(1).index)
bad_dataset.endeffposes_frame = bad_dataset.endeffposes_frame.drop(bad_dataset.endeffposes_frame.tail(1).index)

#init ground truth
jointangles_frame = pd.read_csv("workdir/saveJS.csv")
endeffposes_frame = pd.read_csv("workdir/saveWS.csv")

# import pdb
# pdb.set_trace()

@pytest.fixture(params=[1, 5, 7])
def sample_idx(request):
    return request.param
"""
use fixture to run more tests 

if needed, change the index, or make it to be the full set

there might be some strange syntax limitation with fixture

this comment cannot be in between the above 3 lines of code
"""

def test_item(sample_idx):
	"""
	this test is trying to test if the values are correctly read and stored
	in the dataset from the raw data

	in ``assert`` only scalar not vector equality is checked

	"""

	test_sample = test_dataset[sample_idx]
	true_sample_js = jointangles_frame.iloc[sample_idx, :].values
	true_sample_ws = endeffposes_frame.iloc[sample_idx, :].values
	test_js, test_ws = test_sample['jointspace'], test_sample['workspace']
	assert test_js[0] == true_sample_js[0]
	assert test_ws[0] == true_sample_ws[0]

def test_trans_size(sample_idx):
	"""
	this test is trying to test if the output tensor has the correct shape,
	meaning a certain dimension (1) has to have a certain length (18 or 15)

	this is for practical reasons

	"""
	test_sample = tran_dataset[sample_idx]
	assert test_sample['jointspace'].size()[1] == 18
	assert test_sample['workspace'].size()[1] == 15

def test_read_all():
	"""
	this test is trying to test if all the data has been read in

	this was pushed by coverall because previous test doesn't call __len__()
	the third assertion is for the try catch to see if unmatched data can be handled

	"""
	assert len(tran_dataset) == 64 - 1
	assert len(test_dataset) == 64 - 1
	assert len(bad_dataset) == 0
