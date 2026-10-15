# BrainMirror IXI Kaggle → 私有 GitHub 持久备份

**目的：** 保存已完成批次的模型输入衍生数值与审计记录，防止 Kaggle `/kaggle/working` 会话消失。**不要上传到 public `changxinjiresearch/BrainMirror`**。

## 一次性设置

1. 登录 GitHub，新建一个 **Private** 仓库，建议 `changxinjiresearch/BrainMirror-Data-Private`，创建时添加一个 README，确保默认分支存在。
2. 创建 fine-grained personal access token：仅针对这个 Private 仓库，**Contents: Read and write**。不要复用其他项目的 token，不要在 Notebook 单元格、GitHub Issue 或聊天里粘贴 token。
3. Kaggle Notebook 的 `Add-ons → Secrets` 中创建 `BRAINMIRROR_BACKUP_TOKEN`，值为上述 token，并授予当前 Notebook 使用权限。
4. 确保 Kaggle Internet=On，以便访问 GitHub API。

## 立即备份当前已完成 Shard

当前已经运行的 `ixi_run_all.py` 不会因 GitHub 代码更新而自动改变。**不必取消已有计算。** 如果 Kaggle 有可用的 Terminal（同一运行会话），可在 Terminal 执行下面的 Python 代码，从当前 `/kaggle/working` 上传已完成 Shard。Terminal 与 Notebook 必须共享同一个 `/kaggle/working`；先检查目录。如果终端不可用，等当前计算在批次边界或达到预算停下之后执行。

```python
from kaggle_secrets import UserSecretsClient
from urllib.request import urlretrieve
import os, subprocess, sys

revision = "<PIN_BACKUP_SHA>"
url = (
    "https://raw.githubusercontent.com/"
    f"changxinjiresearch/BrainMirror/{revision}/scripts/ixi_github_backup.py"
)
urlretrieve(url, "/kaggle/working/ixi_github_backup.py")
env = os.environ.copy()
env["BRAINMIRROR_BACKUP_TOKEN"] = UserSecretsClient().get_secret("BRAINMIRROR_BACKUP_TOKEN")
subprocess.run([
    sys.executable, "/kaggle/working/ixi_github_backup.py", "sync",
    "--root", "/kaggle/working",
    "--repo", "changxinjiresearch/BrainMirror-Data-Private",
],env=env,check=True)
```

成功后，程序显示 `BACKUP COMPLETE: uploaded=..., already_verified=...` 和私有 release 链接。**只有看到备份完成、远程校验成功后，才能把相关 Kaggle 结果视为有 GitHub 副本。**

程序会保留：五例试验参考与输出、私有划分清单、已完成 shard 的 aggregate `.npz` 与 receipt `.json`、全量体素检查结果、存在时的最终三维平均/标准差/覆盖图/预览。进行到一半的 shard 会跳过，确保不备份损坏的中间文件。

## 后续自动备份

GitHub 分支中最新版 `ixi_run_all.py` 增加：

```
--backup-repo changxinjiresearch/BrainMirror-Data-Private --require-backup
```

运行前将 `ixi_github_backup.py` 下载至 `/kaggle/working`，并向子进程提供 `BRAINMIRROR_BACKUP_TOKEN` 环境变量。新的运行器会先备份已有结果，再每完成一个 shard 自动备份一次，末尾再同步最终产物。失败会中止后续计算，防止无保护地消耗计算额度。

## 在新 Kaggle 会话中恢复

挂载相同 IXI 私有数据集，安装程序依赖，下载固定版本的 `ixi_github_backup.py`，从 Kaggle Secret 获取 token，然后执行：

```
python /kaggle/working/ixi_github_backup.py restore \
  --repo changxinjiresearch/BrainMirror-Data-Private \
  --root /kaggle/working \
  --release-tag ixi-t1-00f3ec7d490bff06-atlas-v1
```

`release-tag` 必须和 GitHub Release 页面实际显示一致。恢复脚本验证远程文件 SHA256 后写回正确的工作目录，遇到不同内容的已有文件会拒绝覆盖。

## 安全边界

- GitHub 是代码和受限成果的备份位置；Private 不代表绝对匿名化，权限仍需最小化。
- `private` 校验由脚本强制执行；发现目标 repo 是 Public 直接停止。
- GitHub Releases 单文件限制小于 2 GiB；大量小文件不应直接挤入 Git 历史。
- SHARD 完成与上传成功是两个事件。没有远程 Release 校验，就不能称为已备份。
- `private_raw_scans` 应继续保留在原始来源及私有 Kaggle 数据集；本程序备份的是已有已计算产物，而不是复制 581 份原始影像到 GitHub。
- 已完成批次如果只存在会话内而尚未备份，别在保存确认之前关闭 Kaggle 会话。