# BrainMirror 软件架构、API 与数据约束

## 1 架构原则

分为六个逻辑模块：`Web UI`、`MRI Data Service`、`Model Worker`、`Trace Engine`、`Experiment Registry`、`Audit/Export`。当前 P0 只有同机 FastAPI + PyTorch 及静态网页；升级到线上之前必须将可变全局状态改成 session 隔离，并增加认证、权限和独立推理作业。

```
[React + NiiVue + Three.js]  <HTTPS / Auth>  [FastAPI + Session API]
          |                                          |
       (display)                             [GPU Model Worker]
          |                                          |
       [Trace JSON] <--- [Trace Engine: real units/weights/ablations]
                                                     |
                             [Private MRI Storage] [Experiment DB]
                                                     |
                                 [Signed manifests + Git code SHA]
```

## 2 P0 本地文件树

- `brainmirror/model.py`：真实 3D CNN，返回中间激活，支持指定单元消融，卷积连接加权和核验。
- `brainmirror/mri.py`：phantom（仅合成 debug）、NIfTI/NumPy 载入、预处理、受限 ROI warp 和网格生成。
- `brainmirror/server.py`：**只限本地开发** FastAPI；不满足多用户云端安全前提。
- `brainmirror/experiment.py`：新增最小可审计配对试验核心，支持真正前向、真实连接、消融与 provenance。
- `scripts/p0_verify.py`：产生轻量的合成 debug 算术核验 JSON。
- `scripts/p1_inventory_headers.py`：只读取私有目录内的 MRI header 制作数据准备清单（需 nibabel）；不上传数据。
- `web/`：现有单机静态演示及本地 API 客户端；后续重构 React/NiiVue。
- `docs/`：项目科学权威文本、门禁和迭代记录。

## 3 真正的神经元 ID 与边含义

对于 Conv3d：

```
unit_id = model_sha256 + ':' + layer_name + ':c' + channel + ':z' + z + ':y' + y + ':x' + x
edge_type = 'conv3d_weighted_input'
edge = {source_unit_id, target_unit_id, weight, source_activation, contribution, kernel_index}
```

一次观察包括：`x_sha, model_sha, checkpoint_label, layer, unit, pre_activation, post_activation, bias, weighted_sum, tolerance, input_domain, run_id`。偏置单独记录，不用画成神经元。卷积 stride/padding 由实际模型模块决定。

## 4 API 规划

| API | 方法 | 定义 | 状态 |
|---|---|---|---|
| `/api/state` | GET | 当前本地模型、训练来源、层形状与状态 | legacy P0 可用 |
| `/api/analyze` | POST | 真实 forward、输入编辑及被选中单元的真实边/节点 | legacy P0 可用 |
| `/api/train` | POST | 仅本地异步 demo 训练；**不可用于正式高可信科学训练** | legacy P0 可用 |
| `/api/upload` | POST | 本地传入 MRI+ROI；未授权云端禁止开放 | legacy P0 可用 |
| `/api/v1/sessions` | POST | 建独立实验会话，配权限和隔离存储 | P5 |
| `/api/v1/models` | GET | checkpoint 完整身份、可检查 layer 列表 | P3 |
| `/api/v1/runs` | POST | 运行带 provenance 的基线/干预配对实验 | P5 |
| `/api/v1/runs/{id}` | GET | 查询同会话实验状态与结果摘要 | P5 |
| `/api/v1/runs/{id}/units` | GET | 实际节点的分页查询 | P3/P5 |
| `/api/v1/runs/{id}/edges` | GET | 指定单元的实际输入连接 | P3/P5 |
| `/api/v1/runs/{id}/ablate` | POST | 对模型内部选定单元/通道干预 | P5 |
| `/api/v1/checkpoints` | GET | Epoch 快照列表及 hash | P6 |
| `/api/v1/export` | POST | 审计包、轻量数据和清单 | P5 |

所有未来 V1 接口采用明确的输入 schema，权限绑定 session 和用户，禁止任意文件路径读写；大图/large tensor 数据不在单次 JSON 返回中。API 错误包括 INVALID_MASK、MODEL_VERSION_MISMATCH、INTERVENTION_QC_FAILED、DATA_LICENSE_NOT_APPROVED、AUTH_REQUIRED 等。

## 5 运行清单建议

```json
{
  "schema_version": "brainmirror.trace.v1",
  "run_id": "UUID",
  "source_tier": "synthetic_debug | real_healthy_mri",
  "model_sha256": "hex",
  "code_commit": "git SHA",
  "input_sha256": "hex",
  "roi_sha256": "hex",
  "preprocess_sha256": "hex",
  "intervention": {"type":"local_geometry", "parameters":{}, "qc":{}},
  "target_unit": {"layer":"conv3", "channel":0,"z":2,"y":2,"x":2},
  "evidence_level": "L1_REAL_NUMERIC",
  "scientific_validation": "NOT_STARTED",
  "results": {"activation_delta":0, "reconstruction_delta":0},
  "artifacts": [{"path":"...","sha256":"..."}]
}
```

这个示例仅描述 schema，不代表测量值或科学发现。

## 6 安全、隐私与部署

- 本地开发默认 `127.0.0.1`；**禁止**把当前包含上传和训练的老版 `server.py` 直接发布为公网服务。
- 旧版 server 维护进程内**全局** MRI/ROI/model 变量，只适合单用户本地；线上必须改为按用户隔离和后台 worker。
- 对 `*.nii*`, `*.npy`, 原始 mask、`*.pt`、大型 trace 二进制等配置 Git 忽略。公开 Repo 只放源码、虚构 phantom、轻量 QA/摘要和校验清单。
- checkpoint 存储需权限、元数据、许可证审查；不得提交含患者数据的可还原缓存。
- 浏览器缓存和失败日志不得默默保留可识别影像；明确数据删除与过期策略。
- 公开在线预览必须严格标示：`SYNTHETIC ANIMATION` / `REAL PRECOMPUTED TRACE` / `REAL LIVE MODEL RUN`，三者不得混称。

## 7 技术选型依据

PyTorch hooks: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html  
NiiVue: https://niivue.com/docs/  
MONAI: https://project-monai.github.io/MONAI/  
文档版本可能随框架变化，需要在实际实现锁定依赖并复核 API。
