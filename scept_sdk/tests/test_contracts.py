import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from scept import stable_json, tensor_array, tensor_sha, SCEPTError

ROOT=Path(__file__).resolve().parents[2]

def test_exact_js_tensor_sha_parity():
    a=np.array([[1.,-2.5,.25],[4.125,0.,-6.]],dtype=np.float32)
    val=subprocess.run(['node','-e', '''const c=require('./scept-web/core/scept_capture_once.js');(async()=>{const t={type:'float32',dims:[2,3],data:Float32Array.from([1,-2.5,.25,4.125,0,-6])};console.log(JSON.stringify({sha:await c.tensorSha(t),stable:c.stable({id:'standard-v1',config:{seed:7,unit:'MRI'}})}))})().catch(e=>{console.error(e);process.exitCode=1})'''],cwd=ROOT,text=True,capture_output=True,check=True)
    info=json.loads(val.stdout)
    assert tensor_sha(*tensor_array(a))==info['sha']
    assert stable_json({'config':{'unit':'MRI','seed':7},'id':'standard-v1'})==info['stable']

def test_invalid_preprocessing_or_tensor_fails_closed():
    with pytest.raises(SCEPTError,match='UNSUPPORTED_CANONICAL_VALUE'):
        stable_json({'sigma':.0001})
    with pytest.raises(SCEPTError,match='NONFINITE'):
        tensor_array(np.array([float('nan')],dtype=np.float32))
    with pytest.raises(SCEPTError,match='INVALID_SHAPE'):
        tensor_array(np.array(3,dtype=np.float32))
