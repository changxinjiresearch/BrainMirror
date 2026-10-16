# SCEPT · 零云费用部署手册（Oracle Always Free / ARM CPU）

## 状态与费用
- **这是已提交到仓库的部署包，不代表已创建 VM 或通过真实 CPU 跑通模型。**
- 官网：https://www.oracle.com/cloud/free/
- 官方 Always Free 参考：https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm
- **2026 配额：A1 最多 2 OCPU + 12 GB RAM**；只允许 `VM.Standard.A1.Flex` 且必须标记 **Always Free Eligible**，在账号 home region 内。
- 不要升级 Pay As You Go，也不要创建付费存储、负载均衡或计算规格；新账号可能需要银行卡身份认证但不等于订阅。
- 免费 A1 容量可能不足；7 天低利用率实例可能被回收。免费不代表 24/7 SLA。
- 使用 2 CPU、12 GB 内存，不需重新训练，不租 GPU。真实推理+干预可能较慢；未实测前不承诺分钟数。

## 需要用户账户完成的操作
1. 建立 Oracle Cloud Free Tier 账号（登录与卡片验证必须由账户所有者完成）。
2. 在账号 **home region** 新建 Ubuntu `VM.Standard.A1.Flex`，`2 OCPU`、`12 GB`；确认 Free Eligible。不要选择试用期内免费但试用期后收费的配置。
3. 在 VCN Security List 或 NSG 中仅开放 TCP 22（最好限制自己的 IP）、80、443；不用开放 8000。
4. 安装 Docker Engine 和 Docker Compose V2，在 VM 上 clone BrainMirror 的 `phase/scept-transformer-vib-alpha-20261016` 分支。
5. 进入 `services/scept_live/oracle_free`，复制 `.env.example` 为 `.env`。`SCEPT_HOST` 设置为免费域名 `PUBLIC-IP-WITH-HYPHENS.sslip.io`，例如 `192-0-2-10.sslip.io` 是**仅用于文档的保留测试地址，不可直接使用**。sslip.io 是第三方免费 DNS，不保证长期 SLA。
6. 在服务器本地生成访问密钥：`python3 -c 'import secrets; print(secrets.token_urlsafe(48))'`。将输出仅保存在 `.env` 的 `SCEPT_RUN_TOKEN`，`chmod 600 .env`；**不提交到 GitHub，不在 ChatGPT 发送密钥**。
7. `docker compose up -d --build`；端口 80/443 的反向代理由 Caddy 自动尝试免费 TLS 证书。等待模型首次下载（模型权重只在用户发起首个任务时缓存）。
8. 验收 `https://PUBLIC-IP-WITH-HYPHENS.sslip.io/health`（要求 `status=ready`、`cuda=false`、`mode` 对应）；只有确认 HTTPS 后才可配置网站。
9. 将 `web/observatory/scept-config.json` 中的 `api_url` 设为 `https://<SCEPT_HOST>`，重新发布 Vercel 生产网站。网站保留用户在本地输入访问令牌，不写入前端源码。
10. 先用**人工合成的 4 通道 NIfTI** 执行端到端真实模型加载、预测及干预；通过后再上传有合法使用许可的真实 MRI。不要在公用仓库、工单、日志或截图中放个人医疗原始数据。

## 分离与安全约束
- 网页入口仍为 https://brainmirror-observatory.vercel.app 。
- 本服务无用户注册系统，是单人用私有 API，支持 Bearer Token + CORS 但不是多人医疗数据平台，不适合直接开放给陌生人。
- 四模态 NIfTI 上传到你自己的 Oracle VM 临时目录，处理后删除，不经过 GitHub、Kaggle 或 Hugging Face。
- 上传不是端到端本地处理：你必须信任你控制的 VM、安全配置与网络链路。
- 原始 MRI 的数据隐私、临床授权和数据集使用许可仍需核对。
- 不承诺 2 CPU 12 GB 一定足够稳定执行全部干预；如果 OOM，应缩小实验 scope 或仅做可视化回放，绝不悄悄购买额外资源。
- 网站的交互式三维 MRI 仍是局部 voxel preview；其视觉状态记录不是新科学证据。

## 资源与故障
- Compose `api` 限制最大 2 CPU / 10 GB 内存，保留额外内存给 OS/Caddy。
- `docker compose ps` 查看是否健康；`docker compose logs --tail=60 api` 查看不含患者信息的错误类型。
- 令牌仅在 VM 的 `.env`；如果泄露则重新生成后重启容器。
- 不要启用 OCI 免费试用期补贴覆盖的额外付费实例/磁盘。
