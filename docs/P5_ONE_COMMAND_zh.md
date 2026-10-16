# BrainMirror P5：真实 IXI 原生皮层重建，一次运行到完整私有备份

## 已具备的工程

- **从真实 IXI 训练集提取原始分辨率的单人 T1**，已用既有受试者级 SHA256 划分保护 Dev/Test；不得使用 467 人平均脑作为个体皮层输入。
- **通过 Docker 运行官方 FastSurfer 全流程**：分割 + FreeSurfer pial cortical surface reconstruction；要求从 FreeSurfer 免费申请并配置许可。
- **从真实重建目录导出**：两半球 `lh.pial`,`rh.pial` 和各自 `aparc.DKTatlas.annot`，以及 `aseg.mgz` 实际存在的海马、丘脑、杏仁核等深部结构；保留真实 tkrRAS 物理空间。
- **保密、断点保全**：对完整 `surf`、`annot`、`aseg` 文件创建允许列表的压缩备份，对 JSON 表面包及 QC 回执做 SHA256 校验，全部进入批准的 PRIVATE GitHub Release。
- **网站**：浏览器本地导入 `.json.gz`，显示真实个体左右半球、皮层分区、深部结构。标明该个体尚未与 467 人平均脑配准，也未与右侧 AI 联动。

## 严格前提

完整皮层重建不能在当前用户未提供真实计算运行记录前标为「已完成」。本项目也没有可直接控制用户私有 Kaggle Notebook 的已连接执行工具。**Kaggle 浏览器终端并不等同于带 Docker daemon 的 Linux 主机**。请先验证计算环境。

官方 FastSurfer v2.5.4 Docker 参考：`deepmi/fastsurfer:cu128-v2.5.4`（CUDA GPU）和 `deepmi/fastsurfer:cpu-v2.5.4`（CPU，速度慢）。根据 GPU 驱动实际版本选择匹配的镜像，不能盲目假设 Kaggle GPU 支持 CUDA 12.8。FreeSurfer 官方许可免费获取：https://surfer.nmr.mgh.harvard.edu/registration.html 。FastSurfer Docker/本地安装：https://deep-mi.org/FastSurfer/stable/overview/INSTALL.html 。

## Step 1：在 Kaggle 私有环境准备原始分辨率训练受试者

在原来的 Notebook 中（IXI 581 输入和先前的私有 split 清单已经恢复）：

```python
from pathlib import Path
from urllib.request import urlretrieve
import subprocess, sys

src='https://raw.githubusercontent.com/changxinjiresearch/BrainMirror/phase/p5-cortical-reconstruction/scripts/ixi_prepare_native_case.py'
f=Path('/kaggle/working/ixi_prepare_native_case.py')
urlretrieve(src,f)
subprocess.run([sys.executable,str(f), '--manifest','/kaggle/working/brainmirror_ixi_split_v1/PRIVATE_ixi_subject_split_DRAFT.csv', '--data-root','/kaggle/input','--site','GUYS','--out','/kaggle/working/brainmirror_p5_native_case01'],check=True)
```

取得 `/kaggle/working/brainmirror_p5_native_case01/PRIVATE_ixi_case01_native_T1.nii.gz`。这是真实原分辨率的 **一名训练集受试者**，不应上传公开仓库。通过可信的**私人数据传输**放到 Docker 计算主机的 `/data/private/ixi_train_case01.nii.gz`。请检查本人的 IXI 数据使用许可和衍生数据共享限制。

## Step 2：在 Docker 计算主机完成重建与 GitHub 私有备份

**只在 Linux / Docker GPU 主机，且 Docker 可运行时执行。** 计算主机安装 Python 3.10+、`nibabel`、`numpy`、`scipy`、`scikit-image`；获取 P5 分支中的 `scripts/p5_native_runner.py`、`scripts/export_freesurfer_surface.py` 和 `scripts/p5_private_backup.py`，放在同一个 `scripts/` 目录。

FreeSurfer license 请保存到 `/data/private/license.txt`（不能上传任何 GitHub 仓库）。在计算主机的环境中**安全设置** `BRAINMIRROR_BACKUP_TOKEN`，仅授权现有私有仓库 `changxinjiresearch/BrainMirror-Data-Private` 的 `Contents: Read and Write`，绝不打印在终端输出或提交代码。

```bash
python scripts/p5_native_runner.py \
  --t1 /data/private/ixi_train_case01.nii.gz \
  --license /data/private/license.txt \
  --out /data/private/brainmirror_p5_run1 \
  --mode gpu --threads 4 --require-backup
```

GPU 环境不具备匹配的容器能力时，停下检查，不能直接降级后声称已完成；如有合适 CPU 主机，改用 `--mode cpu`，但会明显更慢。

运行完必须出现 **`P5 PRIVATE BACKUP COMPLETE`** 且 `P5_NATIVE_EXPORT_COMPLETE`，Release `ixi-native-p5-case01-v1` 应包含并通过 SHA256 校验：

1. `PRIVATE_p5_native.json.gz`：网页本地可视化网格包；
2. `PRIVATE_p5_native_receipt.json`：来源、哈希、计算版本、审核状态；
3. `PRIVATE_p5_recovery.tar.gz`：只含重建所得允许列表表面与分割，不包含原始 MRI 或许可。

GitHub 可恢复：

```bash
python scripts/p5_private_backup.py restore --root /data/private/brainmirror_p5_run1
```

私有受试者原始 T1 不在备份包内，因此本地原始数据应单独妥善保存。

## Step 3：在 BrainMirror 浏览器检查真实皮层

网址：https://brainmirror-observatory.vercel.app 。展开 `真实 MRI 导入与解剖操作` → `真实高精度皮层及脑区` → 导入 `brainmirror_case01_native.json.gz`（与 Release 中 `PRIVATE_p5_native.json.gz` 同内容）。在 `真实个体` 模式查看：

- 左右半球明显分开，真实皮层沟回可辨认；
- 可单独显示/隐藏左右半球；
- 可选中 `aparc.DKTatlas` 实际标签并在表面高亮；
- 可选中/显示真实 `aseg.mgz` 中存在的海马、丘脑、脑室等结构；
- 显式标记 `UNREVIEWED` 和 **未与 467 人群体模板空间配准**；右侧 `NOT LINKED` 继续保持。

## P5 科学验收门槛（不能用工程测试取代）

1. 用 Freeview 或等效软件把实际 pial/white boundary 覆盖到原始 MRI，检查左右半球、脑沟、脑回与非脑组织伪影；发现失败需记录并重建。
2. 统计每个半球表面顶点、面片，核验标签覆盖比例、pial 与真实图像边界的一致性。
3. 深部结构分割输出检查海马、杏仁核、丘脑、脑室，小脑和脑干是否合理，不用视觉贴图冒充医学标签。
4. 在 ChromeOS/Chrome 亲自进行至少 1 次真实 `.json.gz` 导入、旋转、半球切换、脑区点选及结构隐藏测试。
5. 精细个体皮层 **不能被描述为** 467 人群体均值的真实解剖沟回；科研审核通过前 status 保持 `RECONSTRUCTED_UNREVIEWED`。

以上五项都完成并有审计记录，P5 才允许标记 Complete。后续模型 MRI→AI 真实联动是不同研究阶段，不包含在 P5 的完整定义里。