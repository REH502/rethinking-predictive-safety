# 发行范围和数据隔离

GitHub 发行说明：项目原创贡献采用仓库根目录的 MIT LICENSE；第三方代码、数据和资产仍保留其各自的许可和使用条款。根目录许可不替换下文列出的上游许可。

本包从当前工作树导出三类研究的实际共享实现，保留原生模块布局和必要的版本兼容模块；不是重写的简化算法。相较上一份纯代码包，新增静态输入、安装依赖、配置与验证入口。

包含：HA task YAML、task images、tool catalogs、初始 fixtures；IS-Bench task / BDDL / camera / scene templates、静态元数据、本体和字体；原生源码、相关测试、配置模板、安装工具、文档和 Linux/Python 3.11 依赖 wheels。

排除：`runs/`、历史 `results/` / `reports/` / `traces/`、请求/CoT/token 账本、已有 scores/figures、checkpoint、训练抽取 policy cache、SLURM 日志、私有 `.env`、模型权重、虚拟环境、Git 元数据、job-specific 重复脚本、历史 frozen-source 目录。合成单元测试中的示例任务不是实测数据。

HA 文件依据其现有 MIT LICENSE 保留许可；BDDL 保留自身 LICENSE；Python wheels 内保留上游 license/metadata。其他上游资源沿用各自使用条款；不要把这个研究交付包理解为新的统一开源许可证授权。源参考：[HarnessAudit](https://github.com/eric-ai-lab/HarnessAudit)、[IS-Bench](https://github.com/AI45Lab/IS-Bench)、[BDDL](https://github.com/StanfordVL/bddl)。

生成流程使用显式源目录白名单，拒绝符号链接；逐文件与当前活跃凭据比对、检测私钥/常见真实 token 形式，只输出通过与否；不会把 key 或其哈希写入校验清单。校验清单只记录代码、静态资源及依赖文件的哈希。压缩包在独立临时目录提取后自检，测试产物不会装入包内。

IS-Bench 真正隔离预执行 `environment_h1` 尚未实现；相关规划与禁止启动代码保留。完整代码交付不等于把未实现的研究功能宣称已完成。旧的 checker oracle 不能替代这个功能。
