# 冻结 DER++ 基础检查点门 A，2026-10-29

结果为 NO-GO：两组 TV 为0.004817/0.006118，身份概率差跨组余弦0.3201，未达到预定门。没有训练分支或草案扩展；旧760步基础不足结果不变。

完整设计见 PROTOCOL_ZH.md，结果前哈希见 protocol-freeze.json，结论见 REPORT_ZH.md。8+8视图之前另有预先固定的160批冻结影子校准，用于补足均匀DER++基线本身没有的Q/参考权重；它不是A2ER或原训练轨迹。

run-001保存全部500槽位的输出和评分、影子访问/权重记录、逐槽CSV及独立验证。verify.py用原half-sum单位重算全部校准，以及独立moment表达式复核评分，零新模型前向或更新。所有模型/BN/buffer/历史logit均保持不变。MANIFEST.json记录本目录哈希，prior-record-integrity.json验证旧记录未改。

已完成既定一次运行，不为追门重复运行、换seed或改增强。没有dev/test访问、云GPU、付费、推送或重置卡操作。
