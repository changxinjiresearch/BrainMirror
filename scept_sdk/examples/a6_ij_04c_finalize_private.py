"""04C observational closeout, gated on ALL 594 genuine raw SCEPT tensor replays.

04C-01 summary associations alone never pass full 04C. 04C-02 must first
download and SHA-recheck seven original private 04B raw ZIP shards, verify
all nested observations, and recompute every three-layer tensor summary.
No patient data leaves the authoritative PRIVATE research repo.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import spearmanr
from a6_ij_03b_measure import github,private_contents,API
from a6_ij_03b_archive_chunk_private import commit_once
from a6_ij_04c_association_private import (
    BASE,PRIMARY,L,STATUS as ASSOC_STATUS,canonical,sha,load
)
from a6_ij_04c02_full_tensor_audit_private import STATUS as REPLAY_STATUS

FINAL="PASS_A6_IJ_04C_33_PATIENT_594_GENUINE_ORIGINAL_FULL_TENSOR_OBSERVATIONAL_AUDIT"
OPERATORS=("gaussian_local_sigma3","healthy_pool_median_fill")
CONDITIONS=tuple(f"{o}_dose{d}" for o in OPERATORS for d in (25,50))

def check_finite(value):
    v=float(value)
    if not np.isfinite(v):
        raise RuntimeError("04C full tensor source contains nonfinite value")
    return v

def main():
    session=github()
    # Complete old 04B original source and SHA receipt provenance
    original_patients, original_04b_sha, old_04b, old_receipt=load(session)
    if len(original_patients)!=33:
        raise RuntimeError("Original 33 eligible P0 VAL patients incomplete")
    raw_assoc=private_contents(session,f"{BASE}/analysis.json")
    assoc=json.loads(raw_assoc)
    assoc_receipt=json.loads(private_contents(session,f"{BASE}/archive_receipt.json"))
    if (assoc.get("status")!=ASSOC_STATUS or
        assoc.get("nPatientIndependentUnits")!=33 or
        assoc.get("nOriginalRealSCEPTObservations")!=594 or
        assoc.get("original04BAnalysisFullSHA256")!=original_04b_sha or
        assoc_receipt.get("original04C01AnalysisSHA256")!=sha(raw_assoc) or
        assoc_receipt.get("nOriginalLOPOFolds")!=33):
        raise RuntimeError("Immutable original 04C-01 33-person private analysis is missing/changed")
    original_ids=[r["patient_id"] for r in original_patients]
    groups=[]
    source_proofs=[]
    for chunk in range(1,8):
        p=f"{BASE}/full_tensor/chunk_{chunk:02d}_verified.json"
        raw=private_contents(session,p)
        report=json.loads(raw)
        expected=original_patients[(chunk-1)*5:chunk*5]
        expected_ids=[r["patient_id"] for r in expected]
        source_chunk=old_04b["sourceChunkSha"][chunk-1]
        if (report.get("status")!=REPLAY_STATUS or
            report.get("chunk")!=chunk or
            report.get("patientIds")!=expected_ids or
            report.get("sourceAssetSHA256")!=source_chunk["assetSHA256"] or
            report.get("original04BChunkSHA256")!=source_chunk["sha256"] or
            report.get("nOriginalSourceRunTensorSets")!=len(expected_ids)*18 or
            len(report.get("patientSummaries",[]))!=len(expected_ids)):
            raise RuntimeError("Original complete 04C-02 197x768 tensor chunk replay missing/mismatched")
        for expected_case,patient in zip(expected,report["patientSummaries"]):
            if (patient["patient_id"]!=expected_case["patient_id"] or
                patient["grade"]!=expected_case["grade"] or
                patient["nOriginalRunTensorSets"]!=18 or
                set(patient["modelResults"])!={"S","J"}):
                raise RuntimeError("04C-02 patient identity, grade, S/J original source differs")
            for model in ("S","J"):
                q=patient["modelResults"][model]
                if set(q)!={f"{k}_{a}" for k in CONDITIONS for a in ("lesion","control")}:
                    raise RuntimeError("Original 04C02 eight perturbation model conditions incomplete")
                for condition,val in q.items():
                    if set(val["threeOriginalFullTensorLayers"])!=set(L):
                        raise RuntimeError("Original 04C02 genuine layer observations incomplete")
                    for layer in L:
                        m=val["threeOriginalFullTensorLayers"][layer]
                        if m.get("sourceSummaryNumericallyReverified") is not True or \
                           m.get("patchTokenCount")!=196 or \
                           not (1<=int(m["targetTokenIndex"])<=196):
                            raise RuntimeError("04C original full tensor source reread not verified")
                        for k in ("targetTokenChangedEnergyFraction",
                                  "distalPatchEnergyFraction",
                                  "totalPatchChangeEnergy","clsDeltaL2",
                                  "directTokenDeltaL2","patchRelativeL2"):
                            check_finite(m[k])
                        if (m["targetTokenChangedEnergyFraction"]<0 or
                            m["distalPatchEnergyFraction"]<0 or
                            m["targetTokenChangedEnergyFraction"]+
                            m["distalPatchEnergyFraction"]>1+1e-8):
                            raise RuntimeError("04C original hidden state token-energy fractions invalid")
            groups.append(patient)
        source_proofs.append({
            "chunk":chunk,"originalSHA256":sha(raw),
            "source04BPrivateReleaseFullSHA256":report["sourceAssetSHA256"],
            "nGenuineVerifiedOriginalTensorSets":report["nOriginalSourceRunTensorSets"],
            "nPatients":len(expected_ids)
        })
    if ([r["patient_id"] for r in groups]!=original_ids or len(groups)!=33 or
        len({r["patient_id"] for r in groups})!=33 or
        sum(q["nGenuineVerifiedOriginalTensorSets"] for q in source_proofs)!=594):
        raise RuntimeError("Full 33 original patient and 594 real SCEPT archive-source gate BLOCKED")
    # Spatial-distribution analyses are deliberately secondary to the 04C-01
    # locked TargetToken-vs-Margin observational PRIMARY. Per-patient mean S/J.
    secondary={}
    for operator_dose in CONDITIONS:
        pairs=assoc["privatePatientLevelOutcomeAndRepresentationRecords"][operator_dose]["patientPairs"]
        if [r["patient_id"] for r in pairs]!=original_ids:
            raise RuntimeError("04C-01 original numerical patient row ordering changed")
        decision=np.asarray([float(q["meanTwoModelsOutputContrast"]) for q in pairs])
        result={}
        for layer in L:
            target_fraction=[]
            distal_fraction=[]
            for patient in groups:
                def val(model,arm,key):
                    return check_finite(patient["modelResults"][model][
                        f"{operator_dose}_{arm}"]["threeOriginalFullTensorLayers"][layer][key])
                target_fraction.append(float(np.mean([
                    val(m,"lesion","targetTokenChangedEnergyFraction")-
                    val(m,"control","targetTokenChangedEnergyFraction")
                    for m in ("S","J")])))
                distal_fraction.append(float(np.mean([
                    val(m,"lesion","distalPatchEnergyFraction")-
                    val(m,"control","distalPatchEnergyFraction")
                    for m in ("S","J")])))
            def spearman(xs):
                xs=np.asarray(xs)
                if np.std(xs)<1e-12 or np.std(decision)<1e-12:return None
                c=float(spearmanr(xs,decision).statistic)
                return check_finite(c)
            result[layer]={
              "nOriginalPatients":33,
              "meanTargetedTokenChangeEnergyFractionDifference":float(np.mean(target_fraction)),
              "meanDistalPatchChangeEnergyFractionDifference":float(np.mean(distal_fraction)),
              "targetedFractionVsActualOutputResponseSpearman":spearman(target_fraction),
              "distalFractionVsActualOutputResponseSpearman":spearman(distal_fraction)
            }
        secondary[operator_dose]=result
    result={
      "schema":"scept.a6-ij-04c.complete-original-tensor-observational.v1",
      "status":FINAL,
      "nIndependentOriginalP0ValidationPatients":33,
      "nAuthenticOriginalViTOutputFullTensorReplays":594,
      "nPrivateOriginalSHA256VerifiedReleaseChunks":7,
      "04C01OriginalCompleteAssociationSHA256":sha(raw_assoc),
      "04C02SevenOriginalPrivateReplaySummaries":source_proofs,
      "primaryPreviouslyRegistered04C01":assoc["primaryResult"],
      "original04C01LOPOBaselines":assoc["primaryResult"]["LOPOFixedRidge"],
      "independentOriginalFullTensorSpatialPropagationExploratory":secondary,
      "scientificVerdict":"OBSERVATIONAL_INTERNAL_REPRESENTATION_VS_DECISION_ONLY_NOT_CAUSAL_FAITHFULNESS",
      "otherClaimsUnsupported":[
        "Medical tumor evidence dependency",
        "Direct causal contribution of individual hidden units or circuits",
        "Grad-CAM or CAAP superiority",
        "Independent sealed-test or external generalizability"],
      "sealedP5Test":"NOT_ACCESSED",
      "independentP6":"NOT_ACCESSED",
      "nextStudy":"04D separately preregistered direct activation interventions and robust matched negative controls"
    }
    final=f"{BASE}/final/analysis.json"
    raw=canonical(result)
    check=session.get(f"{API}/contents/{final}",timeout=(30,120))
    if check.status_code==200:
        if private_contents(session,final)!=raw:
            raise RuntimeError("Original private 04C final immutable result conflicts")
        print("04C original complete report already archived; no overwrite",flush=True)
    elif check.status_code==404:
        commit_once(session,final,raw)
    else:
        raise RuntimeError("Original private 04C final destination unreadable")
    receipt={
      "schema":"scept.a6-ij-04c.final-private-receipt.v1",
      "status":FINAL,
      "complete04COriginalReportFullSHA256":sha(raw),
      "original04C01SHA256":sha(raw_assoc),
      "nOriginalPatients":33,
      "nOriginal197x768TensorObservationArchives":594,
      "originalSevenVerified04C02SHA256":source_proofs,
      "noNewNeuralModelInference":True,
      "scientificScope":"OBSERVATIONAL_NONCAUSAL_STUDY"
    }
    commit_once(session,f"{BASE}/final/archive_receipt.json",canonical(receipt))
    print("A6-IJ-04C COMPLETE 33-PATIENT 594-GENUINE-FULL-TENSOR OBSERVATIONAL AUDIT: PASS",flush=True)
    print("Primary Spearman:",assoc["primaryResult"]["primaryRepresentationAssociation"]["observedSpearman"],flush=True)
    print("PRIVATE final:",final,flush=True)

if __name__=="__main__":main()
