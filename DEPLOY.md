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
