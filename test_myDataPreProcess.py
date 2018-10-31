import pytest
import pandas as pd
import myDataPreProcess

# init dataset
test_dataset = myDataPreProcess.ForKinDataset()
#init ground truth
jointangles_frame = pd.read_csv('workdir/saveJS.csv')
endeffposes_frame = pd.read_csv('workdir/saveWS.csv')

@pytest.fixture(params=[1, 5, 7])
def sample_idx(request):
    return request.param

def test_item(sample_idx):
	test_sample = test_dataset[sample_idx]
	true_sample_js = jointangles_frame.iloc[sample_idx, :]
	true_sample_ws = endeffposes_frame.iloc[sample_idx, :]
	test_js, test_ws = test_sample['jointspace'], test_sample['workspace']
	assert test_js[0] == true_sample_js[0]
	assert test_ws[0] == true_sample_ws[0]
