# Fake News Detection - 前期技术探索

## 项目简介

本项目是假新闻检测系统的前期技术探索阶段，重点解决大规模数据处理、特征工程和实验框架搭建等基础问题，为后续深度学习模型开发奠定基础。

### 核心目标
1. **数据基础设施**：处理29.3GB原始数据，创建标准化数据样本
2. **技术方案验证**：测试传统ML方法，验证特征工程方案
3. **实验框架搭建**：建立可扩展、可复现的实验管理系统
4. **资源预测模型**：开发时间预测系统，优化实验规划

## 项目结构

```
fake-news-detection-ml/
├── src/                    # 源代码目录
│   ├── data/              # 数据加载模块
│   │   └── loader.py      # 数据加载器
│   ├── __init__.py        # 包初始化
│   ├── cli.py             # 命令行接口
│   └── experiment_runner.py # 实验运行器
├── data/                  # 数据文件
│   └── FakeNewsCorpus/    # FakeNews数据集样本
│       ├── README.md      # 数据集说明
│       ├── fakenews_test_10.csv      # 测试集（10条）
│       ├── fakenews_tiny_100.csv     # 微型集（100条）
│       └── fakenews_small_1000.csv   # 小型集（1000条，推荐）
├── notebooks/             # Jupyter笔记本
│   └── 01_data_exploration.ipynb  # 数据探索分析
├── models/                # 训练好的模型
│   └── general_models/    # 通用模型
├── docs/                  # 文档
│   ├── 项目技术报告.md    # 中文技术报告（详细）
│   └── PROJECT_TECHNICAL_REPORT.md  # 英文技术报告
├── baseline_model.py      # 基准模型实现
├── final_test_fixed.py    # 完整测试框架
├── regenerate_samples.py  # 数据再生脚本
├── requirements.txt       # Python依赖
├── setup.py              # 包安装配置
└── README.md             # 项目说明（本文件）
```

## 技术特色

### 1. 大规模数据处理
- **流式分层采样**：从29.3GB原始数据中智能采样
- **多规模数据样本**：10/100/1000/5000行标准数据集
- **文本预处理管道**：HTML清理、编码统一、质量验证

### 2. 特征工程框架
- **TF-IDF向量化**：支持n-gram和停用词过滤
- **文本统计特征**：长度、词频、特殊字符比例等
- **可扩展设计**：为深度学习特征预留接口

### 3. 实验管理系统
- **统一实验接口**：标准化实验流程
- **时间预测模型**：基于数据规模的实验时间预估
- **结果记录系统**：完整的实验元数据记录

### 4. 传统ML基准
- **多种算法比较**：逻辑回归、随机森林、SVM、朴素贝叶斯
- **交叉验证评估**：5折交叉验证确保结果稳定性
- **性能基准线**：建立约60%准确率的性能基准

## 快速开始

### 环境配置
```bash
# 创建虚拟环境
python -m venv venv_fakenews

# 激活虚拟环境（Windows）
venv_fakenews\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 数据准备
```python
from src.data.loader import DataLoader

# 加载数据
loader = DataLoader('data/FakeNewsCorpus/fakenews_small_1000.csv')
data = loader.load()

# 查看数据信息
print(f"数据集大小: {len(data)} 行")
print(f"标签分布:\n{data['type'].value_counts()}")
```

### 运行基准实验
```python
from baseline_model import run_baseline_experiment

# 运行基准实验
results = run_baseline_experiment(
    data_path='data/FakeNewsCorpus/fakenews_small_1000.csv',
    model_types=['logistic', 'random_forest', 'svm', 'naive_bayes']
)

print(f"实验结果: {results}")
```

## 技术报告

详细的技术实现和分析请参考：
- **[中文技术报告](docs/项目技术报告.md)** - 完整的前期技术探索报告
- **[英文技术报告](docs/PROJECT_TECHNICAL_REPORT.md)** - 英文版技术报告

## 后续计划

基于前期建立的基础设施，后续工作将聚焦于：

### 深度学习模型开发
1. **Transformer模型实验**：BERT等预训练模型的应用
2. **专用网络架构**：针对假新闻特点的神经网络设计
3. **注意力机制分析**：模型决策过程的可解释性研究

### 特征工程深化
1. **深度语义特征**：基于预训练模型的上下文感知特征
2. **多模态信息融合**：结合元数据和传播特征
3. **领域自适应**：针对假新闻检测的领域特定优化

### 系统化实验
1. **大规模端到端实验**：完整数据集上的系统实验
2. **超参数优化**：系统化的参数搜索和调优
3. **错误分析**：深入分析模型失败案例

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- **项目负责人**：[你的名字]
- **邮箱**：3550124064@qq.com
- **GitHub**：[rikka421](https://github.com/rikka421)

---

**项目状态**：前期技术探索完成 ✅  
**下一阶段**：深度学习模型开发 🚀  
**最后更新**：2026年4月14日