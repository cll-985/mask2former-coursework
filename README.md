# Mask2Former 课程论文复现

[下载固定版本 ZIP](https://github.com/cll-985/mask2former-coursework/archive/refs/tags/v1.0.0.zip) · [模型权重与 SHA-256](https://github.com/cll-985/mask2former-coursework/releases/tag/v1.0.0)

下载并解压 ZIP，用 Python 3.9 或更新版本，在解压目录运行下列命令。数据随 ZIP 提供，Python 自带的标准库即可运行。

```bash
python reproduce.py
```

成功时输出 `REPRODUCTION_PASS`，并生成 `reproduced_results.json`。

| 论文主结果 | 复算值 |
| --- | ---: |
| CarDD 冻结测试集基线 AP（seed3407） | 51.035397 |
| 硬掩码 NMS 融合 AP | 52.765311 |
| AP 增量 | +1.729914 |
| 95% 配对重采样区间 | [0.880189, 2.610238] |

脚本从去除图像身份的匹配事件重建精确率和召回率，并按 COCO 的 101 点插值计算主 AP。从归档的 2,000 次抽样重算区间，并由 TP、FP、真值数和图像数重算论文第 4.3 节的八项校准配置。第 4.4 节的几何诊断提供归档 AP 及差值核对。`results/` 还包含其他测试对照和候选覆盖汇总。

## 权重和重新推理

Release 提供 seed3407 和 seed42 两份最终权重。前者是基线，融合方法使用两份权重和 `source/offline_tools/compare_fusion_rules.py` 中的硬掩码 NMS。源代码和配置见 `source/`。

若要从原始图像重新推理和做 COCO 掩码匹配，按 [CarDD 官方许可](https://cardd-ustc.github.io/docs/CarDD_license.pdf) 从 [作者项目](https://cardd-ustc.github.io/) 取得数据，准备 Linux、CUDA、Detectron2 和 MSDeformAttn 环境，按 `source/INSTALL.md` 配置依赖。将两份权重下载到 `weights/`，然后运行：

```bash
python reproduction/run_frozen_inference.py --project-root source --data-root "/path/to/CarDD_COCO" --weights-dir weights --output-dir verification/gpu_inference
```

该入口检查两份权重、冻结配置和测试标注的 SHA-256。E08、E09、其他验证集对照及区域编辑的原图、真值和完整匹配输入需从对应的数据归档取得。随包校准复算使用归档计数，几何部分核对归档 AP 差值。数据许可约束的原图、真值和论文中的内部裁剪图未随仓库分发。
