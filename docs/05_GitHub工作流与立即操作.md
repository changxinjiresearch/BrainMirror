# BrainMirror — GitHub 初始化、提交、协作与安全操作

## 0. 当前真实状态

已验证连接的 GitHub 账号为 `changxinjiresearch`。截至准备此交付时，现有可访问的仓库中未发现独立 `BrainMirror` 仓库。当前已连接的 GitHub 操作具备**向已有仓库新增文件/提交**的能力，但没有新建仓库接口。因此本交付先包含带首次 commit 的本地 Git 项目和 git bundle；远程 GitHub 创建之前不能宣称上传成功。

## 1. 新建远程仓库（一次性）

直接打开：<https://github.com/new?name=BrainMirror>

填写：
- Owner：`changxinjiresearch`
- Repository：`BrainMirror`
- Description：`Anatomy-grounded interactive observatory for authentic neural network mechanisms`
- Visibility：**Private（建议初始为私有，科学源代码稳定后可再决定是否公开）**
- **不要**勾选初始化 README / .gitignore / LICENSE（本地仓库已有首个 commit）。

创建后远程应为 `https://github.com/changxinjiresearch/BrainMirror`，但这个 URL **只有创建操作返回成功后才是存在的仓库**。

## 2. 推送已有本地 git 历史（在能使用 git 的终端）

```bash
cd BrainMirror
git remote add origin https://github.com/changxinjiresearch/BrainMirror.git
git push -u origin main
```

或在 ChatGPT 已能访问新仓库后，利用 GitHub 连接中的 `create_file`/`create_tree`/`create_commit`/`update_ref` 操作提交。这种方式可以避免用户本地安装 git，但需要远程仓库先存在。

## 3. Branch & PR

- `main`：稳定可复现。启用需要 CI PASS 的分支保护。
- `phase/p0-foundation`, `phase/p1-data-governance`, `phase/p2-real-mri-learning`, `phase/p3-tracing`, `phase/p4-mri-editor`...
- 每项变更关联独立 issue：例如 `[P1.03] Build subject-level split manifest`。
- 一个 PR 专注一个能力，提交“代码 + 测试 + 报告 + 失败说明”，不只贴截图。
- 主分支不保留不可审计的 phantom 成功率“科研结论”。

## 4. GitHub Issues 初始队列

| Issue | 标题 | 依赖 | 完成标准 |
|---|---|---|---|
| #P0-01 | Verify exact Conv3d edges on CPU | 无 | pytest PASS，边界位置覆盖 |
| #P0-02 | Provenance manifest schema & receipts | P0-01 | 每次 trace 写源类型、model hash、input hash |
| #P0-03 | Remove inherited demo assumptions | P0-02 | 代码与 UI 不把 synthetic 当真实 MRI |
| #P1-01 | MRI data source license and cohort approval | P0 | 许可与审计证据 |
| #P1-02 | Header QC, affine, spacing and split | P1-01 | manifest 和无泄漏报告 |
| #P2-01 | Training baseline and frozen-probe checkpoints | P1 | ≥2 checkpoint 的真实数据运行记录 |
| #P3-01 | Stable raw-unit/weighted-edge API | P2 | 数值QA+真实张量追溯 |
| #P4-01 | NiiVue real MRI workstation | P1 | 体素与 affine 一致性 |
| #P5-01 | Private session-scoped inference | P3,P4 | 多用户隔离/安全检查 |
| #P7-01 | Fixed anatomy intervention protocol | P1–P5 | 锁定实验、对照与指标 |

这是一份 Issue 准备清单，尚不是 GitHub 上已创建的 Issue。

## 5. 合并前安全清单

- 不出现 `.nii`, `.nii.gz`, MRI DICOM、原始 `.npy`, 含可识别病人 ID 的 JSON。
- 不出现 GitHub token、Kaggle key、API 凭证、`DATABASE_URL` 或 `.env`。
- trace 必须声明 synthetic / real / precomputed / live。
- 模型权重和大型结果须使用经授权存储；SHA manifest 是链接，不是私密数据的公开许可。
- README 中每项 `PASS` 有测试执行记录、运行环境与 commit。

## 6. Git bundle 的用法

交付包另外附 `BrainMirror-initial.bundle`，其中是**已创建的本地 Git 首次提交历史**；若要恢复：

```bash
git clone BrainMirror-initial.bundle BrainMirror
git -C BrainMirror switch main
```

仓库中的历史标签仅描述工程实现，**不**代表正式科学结果已获验证。
