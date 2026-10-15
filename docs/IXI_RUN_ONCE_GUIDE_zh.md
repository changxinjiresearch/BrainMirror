# BrainMirror IXI T1：467 训练受试者的自动三维群体脑候选模型

本工具处理真实健康人脑解剖 MRI。**不包含人脑神经元电活动、认知、动力学全脑仿真。**

## 已存在的 Kaggle 私有输入

- `/kaggle/input/datasets/.../brainmirror-ixi-t1-raw`: 581 个原始 T1 NIfTI。
- `/kaggle/working/brainmirror_ixi_split_v1/PRIVATE_ixi_subject_split_DRAFT.csv`: 467 Train / 57 Dev / 57 Test；草案 SHA256 `00f3ec7d490bff06f529640ccfed4bbae02f172c2f92f4e851e756cc1cf4e40a`。
- `/kaggle/working/brainmirror_ixi_pilot5_v1/case01/{brain.nii.gz,mask.nii.gz}`: 五例试验参考脑。
- `/kaggle/working/brainmirror_ixi_pilot/{mri_synthstrip,synthstrip.1.pt}`: 原已下载的 SynthStrip 代码与模型权重。

如这些 `/kaggle/working` 文件已因会话重启而丢失，须先从**私有持久输出/数据集**恢复。不可只剩原始 IXI 数据便直接运行。

## Kaggle 一次启动

```python
%pip install -q nibabel SimpleITK surfa scipy numpy

from pathlib import Path
from urllib.request import urlretrieve
import subprocess, sys

base = 'https://raw.githubusercontent.com/changxinjiresearch/BrainMirror/<PINNED_COMMIT>/scripts/'
for name in ['ixi_full_atlas.py', 'ixi_run_all.py']:
    urlretrieve(base + name, '/kaggle/working/' + name)
# exporter is in anatomy branch, download separately if approved private preview is desired
urlretrieve('https://raw.githubusercontent.com/changxinjiresearch/BrainMirror/e8e1b57eb5213b1f50f7402c20884f3ce944f286/scripts/export_anatomy_preview.py', '/kaggle/working/export_anatomy_preview.py')

result = subprocess.run([
    sys.executable, '/kaggle/working/ixi_run_all.py',
    '--expected-manifest-sha256', '00f3ec7d490bff06f529640ccfed4bbae02f172c2f92f4e851e756cc1cf4e40a',
    '--max-minutes', '480', '--make-private-preview',
])
print('Run return code:', result.returncode)
```

在冻结 GitHub 源码提交后，将 `<PINNED_COMMIT>` 替换为实际提交哈希，勿使用随时间变化的 `main` 文件。

## 运行过程与产物

1. 一次性完成全队列真实体素/空间有效性审计，若损坏则中止并记录于私有 QC 文件。
2. 只使用 467 Train；按每批 12 例依次执行 SynthStrip、仿射配准、掩膜 Dice >=0.70、归一化，并累加平均的分子/分母与方差统计量。
3. 每批独立输出 `shard_NNN/PRIVATE_aggregates.npz` 和 `PRIVATE_shard_receipt.json`，并通过清单 SHA 和参考模型 SHA 实现可重复/可续跑。
4. 全部约 39 批完成后，自动生成 `final/ixi_train_affine_mean.nii.gz`, `ixi_train_affine_sd.nii.gz`, `ixi_train_coverage.nii.gz`。
5. 若指定 `--make-private-preview`，会生成用于浏览器本地导入的**私有** `final/PRIVATE_BrainMirror_IXI_preview.json`。不能在科学和许可审核前上传到公开 GitHub/Vercel。

## 中断 / 续跑 / Kaggle 持久化

- 程序停止后重跑相同命令，会验证已完成批次的输入指纹并跳过，继续缺失批次。
- **如果断电/强制中止发生在某个批次内部**，该 `shard_NNN` 可能是半成品；程序会拒绝覆盖。应将该目录改名为 `shard_NNN.incomplete` 以保留取证，再重新运行。
- **Kaggle `/kaggle/working` 不是跨会话自动保留的仓库**：请在会话终止前使用 Kaggle 输出版本功能，并在 Version Outputs 中核验实际生成的批次文件；`Save & Run All` 可能重新运行 Notebook。若超过单次会话配额，需要将已经完成的批次保存为 Kaggle 私有持久输出/数据集并在新会话恢复，否则计算不能续跑。
- 无人连接 Kaggle 账号的本助手**不能代替 Notebook 执行 GPU/CPU、保证超时前保存或声明已计算完 467 例**。

## 科学准出

`P1_PASS=false` 和 `P2_TRAINING_AUTHORIZED=false` 保持至独立审核完成。仿射单参考平均不是无偏非线性 atlas，Dice 也不能代替局部脑解剖、脑区分割、采集中心偏差和眼见 QC。历史五例试验可作为工程参考，不应将其写成最终研究结论。