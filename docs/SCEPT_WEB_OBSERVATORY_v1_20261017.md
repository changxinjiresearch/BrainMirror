# SCEPT Web Observatory v1 · 部署与科学边界

## 已部署位置
- 生产站点：https://brainmirror-observatory.vercel.app/
- 代码：`web/observatory/index.html`、`style.css`、`app.js`
- Vercel 项目：`brainmirror-observatory`
- 2026-10-17 更新为前端静态研究工作台。网站版本通过 Vercel source-files 部署，**尚未**通过 GitHub 自动化发布；仓库中的开发分支为 `phase/scept-transformer-vib-alpha-20261016`。

## 已实装、可直接使用
1. 左侧从用户设备读取 3D NIfTI-1 (.nii/.nii.gz)，支持真实轴向切片及简单的 3D 体素点云预览；可叠加同维度的 ROI 掩膜。文件仅保留在浏览器内存，不自动上传。预览停留在 voxel space，不声称临床物理空间配准或任意旋转高精度体绘制。
2. 右侧渲染官方 SwinUNETR 的阶段结构（2 blocks per stage、3/6/12/24 heads），可点击选择。此处只是**真实模型的架构拓扑**，灰色结构边不是实测的神经元级连接；导入 SCEPT JSON 后，在路径页展示记录中的原子级模块、window/head/query/key 和候选原子顺序。
3. 支持 Kaggle 实验输出 JSON `scept_fast_result.json` 和 `scept_auto_result.json` 的本地导入（根据实际字段识别）。全程通过浏览器，无需 GPU 也无需 Kaggle 会话。
4. 默认显示从用户提供的 Kaggle 运行截图转录的已知初次结果：BraTS20_Training_002、WT 0.9041616723036165、TC 0.9479754601226994、ET 0.8629651506202008、干预/恢复均 0、14.25 秒。**不是已经导入的官方原始 JSON**；必须明确显示“截图转录”。
5. 保存本次会话的视觉选项/实验事件，支持导出审计 JSON 和当前 MRI 画布 PNG。视觉快照不等于新增实验数据，审计包不包含原始 MRI 体素。
6. 新导入的 MRI 不得自动认定与当前实验 JSON 为同一个患者；暂时一律标记“未配对实验”。

## 未实现 / 不得宣称完成
- **在线 GPU 工作服务尚未部署**。网站的“新实验”按钮明确解释缺少实时计算后端，不能伪造 SwinUNETR 输出。
- 干预后逐节点实时动画与真实逐原子激活流、在线自动路径搜索、服务端实验数据库、身份鉴权、MRI 处理工单和自动持久化尚未完成。
- 预训练模型不是 Kaggle 训练产物；目前使用 MONAI 官方 BraTS21 Fold1 权重。BraTS21/BraTS20 病例重叠未核查；局部 Patch Dice **不得**当作患者独立泛化验证。
- Kaggle 当前仍是既有一次实验的执行环境；下一步由独立、受鉴权的 GPU worker 取代，不再要求用户通过 Kaggle UI 操作。

## 接下来的受保护 GPU 服务边界
- 后端直接加载校验 SHA-256 的官方 SwinUNETR 权重。**无需从零训练**，但真正规模的 3D 推理和实验干预仍需要计算资源。
- 服务端需要 session/鉴权、资源限制和审计，不能把旧版 `brainmirror/server.py` 带有全局状态与上传接口的演示直接公网暴露。
- 新实验请求需要清楚区分：当前 MRI hash、checkpoint hash、预处理版本、ROI/输入 QC、模块原子坐标、实测激活/干预效应、失败与负对照、实验时间、可复现实验报告。
- GPU worker 未运行时，网站只提供本地载入、绘制、已存记录回放与导出；不返回虚构推理数据。

## 验收
- GitHub 源代码通过 JavaScript 语法编译检查，DOM ID 引用完整且无重复。
- Vercel 最新部署为 READY、生产别名已绑定 `brainmirror-observatory.vercel.app`。
- 外部网页探测因为当前工具端访问策略而未能执行浏览器端到端测试；应在实际浏览器中核查 URL、导入 JSON、NIfTI 预览与下载。
