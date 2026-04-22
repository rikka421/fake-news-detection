# 前端部署指南 — Fake News Detection API

## 前置要求

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) 已安装并运行

---

## 快速启动（推荐）

```bash
# 1. 克隆仓库（或直接拿到项目文件夹）
git clone https://github.com/rikka421/fake-news-detection.git
cd fake-news-detection

# 2. 一键启动
docker compose up --build
```

服务启动后访问 `http://localhost:8000`，无需任何额外配置。  
**不需要重新训练**，模型文件已包含在仓库中。

---

## 接口使用

### 检查服务状态

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### 预测单条新闻类别

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Scientists discover new planet", "content": "Researchers at NASA..."}'
```

**返回示例：**
```json
{"label": "reliable"}
```

### 批量预测

```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Breaking news headline", "Another article text"]}'
```

**返回示例：**
```json
{"labels": ["clickbait", "junksci"]}
```

### 查看模型信息

```bash
curl http://localhost:8000/model-info
```

---

## 接口完整文档

启动后访问：`http://localhost:8000/docs`（Swagger UI，可直接在浏览器中测试）

---

## LLM 实验结论（前端部署相关）

当前仓库已做 `Qwen/Qwen2.5-0.5B-Instruct` 的零样本评测（CPU 与 GPU 环境）：

- CPU（`torch==2.11.0+cpu`）:
  - `generation` 模式：单条约 `7.81s`，10k 预计约 `21.69h`
  - `likelihood` 模式：单条约 `181.03s`，10k 预计约 `502.87h`
- GPU（`torch==2.11.0+cu126`, RTX 3050, FP16）:
  - 修复长文本截断策略后（保留 `Label:` 指令），`generation` 200 条随机样本：平均 `0.588s/条`
  - 吞吐约 `1.70 条/秒`
  - 10k 预计约 `98.08 分钟`
  - 标签提取命中率约 `87.5%`
  - 分类效果仍较弱：Accuracy `0.070`，Macro F1 `0.0469`

结论：修复评测流程后，当前 GPU 条件下速度仍难稳定满足“10k 数据约 1 小时”目标，且分类效果明显不足，不建议直接替代现有线上模型。

前端部署建议：

- 在线接口默认使用当前 `SVM + TF-IDF` 服务（`/predict`、`/predict-batch`）
- LLM 作为离线实验能力保留，用于后续 prompt 与少样本策略探索
- 若必须在线使用 LLM，请固定 GPU 推理实例并增加批处理/并发限流

---

## 可能的类别标签

| 标签 | 含义 |
|------|------|
| `reliable` | 可信新闻 |
| `clickbait` | 标题党 |
| `conspiracy` | 阴谋论 |
| `junksci` | 垃圾科学 |
| `bias` | 带有偏见 |
| `satire` | 讽刺/戏谑 |
| `hate` | 仇恨内容 |
| `unreliable` | 不可信 |

---

## 停止服务

```bash
docker compose down
```

---

## 仅用 Docker（不用 Compose）

```bash
docker build -t fakenews-api .
docker run -p 8000:8000 fakenews-api
```
