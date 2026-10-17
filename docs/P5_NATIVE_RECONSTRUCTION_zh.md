# BrainMirror P5 — 真实皮层重建与可交互脑区：科研实施说明

## 科学定义

- **已完成**：467 人 IXI T1 训练队列的 **2 mm 仿射群体均值**。其 96×96×71 网页预览能供群体 MRI 切片和粗表面，但**不能恢复已经丢失的个体脑沟/脑回**。保留为左侧 `群体参考` 模式。
- **P5 目标**：从 **一份未经群体平均的真实 IXI 个体原始 T1 MRI** 运行 FreeSurfer/FastSurfer 皮层重建，使用 `lh.pial / rh.pial`、对应的 `aparc*.annot`，以及 `aseg.mgz`，导出可交互的左右半球、皮层标签与少数经独立分割的深部结构。
- **科研边界**：**个体重建不等于 467 人群体解剖表面**。当前个体皮层与群体平均 MRI 未进行共同空间配准，网页不得叠加它们后宣称空间对齐；更不等于已经与右侧 AI 模型实时联动。

## 工具与资质

推荐 FreeSurfer `recon-all -all` 或 FastSurfer（先分割后表面）。运行完整表面重建**需要 FreeSurfer 免费注册许可**；FastSurfer `recon-surf` 同样要求该许可。详见官方文档：
- https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
- https://deep-mi.org/FastSurfer/dev/scripts/recon_surf.html
- https://github.com/Deep-MI/FastSurfer/blob/dev/doc/scripts/RUN_FASTSURFER.md

Kaggle 对 Docker/系统级安装和会话资源存在限制，不能假设 `docker run` 可以在 Kaggle Notebook 里直接执行。推荐能运行 Docker 的 Linux GPU 主机/受控计算环境，或专门为 Kaggle 打包并验证原生 FastSurfer 安装后再执行。

## 首次科学运行（先完成 1 例，不是 467 例）

1. 从之前已锁定的 **Train** 受试者中选择 1 份合格、原始分辨率 IXI T1；不能把 
   `ixi_train_affine_mean.nii.gz` 当作输入。
2. 准备 FreeSurfer `license.txt`，**不要提交 GitHub，也不要贴在聊天**。
3. 使用经验证的 FastSurfer 安装，运行官方 `run_fastsurfer.sh`，示例：

```bash
./run_fastsurfer.sh \
  --t1 /data/ixi_train_native_t1.nii \
  --sid brainmirror_case01 \
  --sd /results/freesurfer_subjects \
  --fs_license /private/license.txt \
  --parallel
```

若使用 FreeSurfer 官方流程：

```bash
export SUBJECTS_DIR=/results/freesurfer_subjects
recon-all -i /data/ixi_train_native_t1.nii \
  -s brainmirror_case01 -all -openmp 4
```

这一步涉及较大环境依赖和数小时级计算，不应在用户没有运行结果时标记为完成。

## 导出私有三维解剖

需要 Python `numpy`, `nibabel`, `scipy`, `scikit-image`：

```bash
python scripts/export_freesurfer_surface.py \
  --subject-dir /results/freesurfer_subjects/brainmirror_case01 \
  --annotation aparc.DKTatlas \
  --out /results/private/brainmirror_case01_native.json.gz
```

如果工具只产生 `lh.aparc.annot/rh.aparc.annot`，将 `--annotation` 改为 `aparc`；同一受试者的 pial/注释必须成对存在。输出使用 **FreeSurfer tkregister RAS 毫米物理空间**；`aseg.mgz` 也按同一受试者的 `vox2ras_tkr` 转换，不允许凭视觉随意移动网格。

## 质量门槛

- 每个半球表面都有大量有效顶点和三角形；几何有限、索引合法、非退化比例可接受。
- `lh/rh` 表面和注释点数一致；真实 ROI 标签映射表来自 `.annot`，不能预设某些点代表海马体。
- 深部结构来自原生 `aseg.mgz` 的原始解剖 label ID，选择显示仅代表该受试者的分割，不是手工放置。
- 检查皮层、白质边界、左右半球、脑干与主要深部结构，以及分割失败/异常情况。
- 网页只在用户本地浏览器中导入私有 `.json.gz`，不会在加载后网络上传。**绝不能把这个个体衍生文件直接发布到公开 GitHub 或 Vercel**。
- 右侧模型 `NOT LINKED` 状态保持不变，直到部署真实 MRI 模型和因果机制解释管线。

## 交互验收

- `群体参考`：显示 467 人 MRI 群体参考与真实方向切片。
- `真实个体`：手动导入 `brainmirror_case01_native.json.gz` 后，显示左右半球清晰脑沟/脑回；能独立隐藏半球，开启或隐藏 3D 深部结构。
- 选择或点击真实皮层区域，按对应 FreeSurfer 注释显示区域名称与高亮；点击深部结构显示其来自 `aseg` 的 ID 与名称。
- 网站**明确标注**该个体表面与 467 人群体平均 MRI 切面尚未配准，不能把二者当作空间对应结果。

## GitHub 存储策略

**公开 BrainMirror 代码仓库**：导出与加载程序、测试、配置与科研方法。

**私有数据仓库**：仅在完成该受试者衍生数据使用许可与隐私核查后备份 `.json.gz`、分割指标、QC 截图；严格使用单独授权的结果仓库和校验值。不能对含个体 MRI 的文件沿用允许发布 467 人均值预览的公开许可决定。