# 当前交接：Pets 原始归档缺失

先读 `REPORT_ZH.md`、`STATUS.json`、`PROTOCOL_ZH.md`。当前不运行 GPU；本轮原始截止是 2026-10-31 01:14 UTC，不能重置。

所有命令从项目根目录运行，解释器使用已有 `.venv/bin/python`；依赖仅位于研究目录 `deps/`，由 `core.py` 添加到导入路径。不要升级依赖或绕过 `weights_only=True`。

可信原始数据恢复路径：

1. 两份归档须同时符合 `REPORT_ZH.md` 中 TFDS SHA256、torchvision MD5 和尺寸。重打包 zip/tar、不同 split、预处理后的镜像均不能替代。新下载的本任务归档应放入研究目录 `assets/`，纳入 3 GiB 总账；若用历史合法缓存，可直接只读导入。
2. 使用 `import_archives.py --images <绝对路径> --annotations <绝对路径>`，它先校验两个归档再安全提取。仅当归档确为本任务生成且位于本目录 `assets/` 时，才可增加 `--remove-task-created-archives`，以核验后清理压缩副本。不要删除历史缓存。若输入是本任务新生成、但留在其它目录的归档，其字节也必须计入新增峰值，不能靠目录位置排除预算。
3. 运行 `.venv/bin/python -B research-view-route-20261030/manifest.py`，检查真实 `data-validation.json`。`fixtures/` 的合成数据验证不能代替此步。
4. 运行 `.venv/bin/python -B research-view-route-20261030/supervise.py pilot.py preflight`。这会计入已有 MPS 总账，完成 3 个 LP 和丢弃副本校准；不算 val/test 效用。必须检查 `preflight.json` 通过、完整 24 runs 预测在剩余预算内，且 `protocol.lock.json` 与 hash 已生成。
5. 只有通过第 4 步后才运行 `.venv/bin/python -B research-view-route-20261030/supervise.py pilot.py formal`。它按固定顺序完成所有 24 个 run、锁定全局 B*/C* 后仅评估 6 模型 test。不因中途 val 改 N、顺序、臂、LR 或门。

恢复槽保存 model、optimizer、schedule 位置、CPU/MPS/Python RNG、stateless 数据/增强/routing 位置、Ent cache 和累计时间。遇到外部中断且仍有完整预算时，可使用同一监督器加 `pilot.py formal --resume`，全局最多一次；只恢复 hash 读回验证的槽，不静默重跑已有 run。OOM、不支持算子、资源不足不能通过 resume 换设备或缩规格救。该恢复代码已保存，但尚无真实 run 可以验证恢复流程。

`supervise.py` 持续累计整个 MPS 工作进程时间，不仅计训练 kernel；加载、增强、切换、等待和保存均在阶段内。进程失败也不会清零。若监督器本身异常留下 active 记录，应先核查原 PID 和实际结束时间并保守补记账本，不能直接清除 active 再开新进程。

当前各文件的 SHA256 和字节数见 `delivery-manifest.json`。该清单不包含自身；不需要运行 pip、访问数据源或模型推理来核对文件。
