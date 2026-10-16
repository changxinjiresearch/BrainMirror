import numpy as np
import pytest
from scripts.scept_pretrained_brats21_fast import MODALITIES,PREDICTED,normalize

def test_official_modalities_and_output_labels():
    assert MODALITIES==("flair","t1ce","t1","t2")
    assert PREDICTED==("TC","WT","ET")

def test_nonzero_channelwise_zscore():
    x=np.zeros((6,6,6),np.float32)
    x[:,:,:3]=np.arange(108).reshape(6,6,3)+1
    y=normalize(x)
    assert not y[:,:,3:].any()
    assert np.mean(y[x!=0])==pytest.approx(0,abs=1e-6)
    assert np.std(y[x!=0])==pytest.approx(1,abs=1e-6)
    with pytest.raises(ValueError):normalize(np.zeros((6,6,6),np.float32))
