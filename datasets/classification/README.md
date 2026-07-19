# 路由分类数据

建议以“该图片最适合交给哪个下游模型”为唯一分类依据。判定优先级为：跑道细节足够、机场全貌、云与天气系统、其他。

`capture_domain` 建议至少区分 `clean_rgb`、`simulated_sstv` 和 `real_mmsstv`。最终测试集必须包含真实 MMSSTV 解码图片。
