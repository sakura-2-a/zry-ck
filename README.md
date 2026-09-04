# 基于深度学习的轴承表面缺陷智能检测系统

《制造智能技术》课程设计 · 完整可运行的 B/S 架构 demo

采用 KolektorSDD 公开表面缺陷数据集，融合三个技术方向形成两级检测流水线：
**YOLOv8**（计算机视觉：缺陷区域定位）→ **SVM**（机器学习：二级判别）→ **遗传算法**（智能优化：超参数寻优）

## 技术栈

| 层级 | 技术 | 角色 |
|---|---|---|
| 算法方向一 | YOLOv8（ultralytics） | 缺陷区域定位（一级检测） |
| 算法方向二 | SVM（scikit-learn） | 缺陷/误检二级判别 |
| 算法方向三 | 遗传算法（手写实现） | SVM/检测超参数优化 |
| 后端 | FastAPI + SQLAlchemy 2 | API 服务与业务逻辑 |
| 前端 | Vue3 + Vite + Element Plus + ECharts | 检测/历史/统计/模型信息页面 |
| 数据 | PostgreSQL + Redis | 检测记录持久化 / 结果缓存 |
| 部署 | Docker + docker-compose | 一键部署 |

## 目录结构

```
轴承缺陷项目/
├── docs/SPEC.md            # 规格说明书（设计蓝图，开发前必读）
├── backend/
│   ├── app/                # FastAPI 后端应用
│   │   ├── api/            # 路由层：detect / records / stats / models
│   │   ├── core/           # 核心层：配置 / 数据库 / Redis 缓存
│   │   ├── models/         # SQLAlchemy ORM 模型（3 张表）
│   │   ├── schemas/        # Pydantic 请求/响应模型
│   │   └── services/       # 检测流水线（YOLO + SVM 两级）
│   ├── scripts/            # 离线脚本：数据准备 / 训练 / 优化 / 评测 / 种子导入
│   ├── tests/              # pytest 单元测试
│   ├── weights/            # YOLO 权重（.pt）
│   ├── artifacts/          # SVM 模型 / 特征缓存 / GA 结果 / 评估图表
│   └── runs/               # YOLO 训练输出
├── frontend/               # Vue3 前端（检测/历史/统计/模型信息四页面）
├── datasets/kolektor_sdd/  # 数据集（原始数据 + YOLO 格式转换结果）
├── docker-compose.yml      # 一键部署编排（PG + Redis + 后端 + 前端）
├── venv/                   # 虚拟环境（M0 已创建）
└── requirements.txt        # Python 依赖清单
```

## 快速开始

### 1. 激活虚拟环境（已创建）

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1
# 若提示"禁止运行脚本"，先执行：Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # 应输出 True
```

> 国内加速：任意 pip 命令后加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`

### 3. 训练流水线（按顺序执行）

```bash
python backend/scripts/prepare_dataset.py        # M1 数据准备（掩码→bbox、划分）
python backend/scripts/eda.py                    # M1 EDA 可视化
python backend/scripts/train_yolo.py             # M2 YOLO 训练（GPU 约 2~5 分钟）
python backend/scripts/eval_yolo.py              # M2 官方 Test 集评估
python backend/scripts/extract_features.py       # M3 候选框特征提取缓存
python backend/scripts/train_svm.py              # M3 SVM 基线训练
python backend/scripts/ga_optimize.py --mode all --n-jobs 16  # M4 GA 优化（多核并行）+ 随机/网格对比
python backend/scripts/eval_pipeline.py          # 图像级流水线评测（val+test）
```

产物：模型权重在 `backend/weights/`，评估指标/图表/GA 结果在 `backend/artifacts/`。

### 4. 启动应用（本机便携版，无需 Docker）

```bash
# ① PostgreSQL（已初始化于 C:\pg_bearing\data，用户名 postgres 无密码）
C:\pg_bearing\pgsql\pgsql\bin\pg_ctl.exe -D C:\pg_bearing\data -l C:\pg_bearing\pg.log start

# ② Redis（tools/redis 便携版）
tools\redis\redis-server.exe --port 6379 --bind 127.0.0.1

# ③ 种子数据导入（把离线训练指标写入数据库，供模型信息页展示）
python backend/scripts/seed_models.py

# ④ 后端（backend 目录下）
cd backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# ⑤ 前端（frontend 目录下，另开终端）
cd frontend
npm install --registry=https://registry.npmmirror.com
npm run dev        # 访问 http://localhost:5173
```

> 后端接口文档：http://localhost:8000/docs

### 5. Docker 一键部署（可选，需已产出训练模型）

```bash
docker compose up -d --build
# 前端 http://localhost:8080   后端 http://localhost:8000
```

## 模块进度

| 模块 | 内容 | 状态 |
|---|---|---|
| M0 | 项目规划与规格说明 | ✅ 已完成 |
| M1 | 数据准备（掩码→bbox、EDA、数据划分） | ✅ 已验证 |
| M2 | YOLOv8 训练 + Test 评估 | ✅ 已完成 |
| M3 | SVM 特征提取与训练 | ✅ 已完成（含 3 次修订：手写 HOG / 低阈值候选 / 负样本增强） |
| M4 | 遗传算法超参数优化 | 🔄 GA 多核并行运行中（完成后自动固化模型并重启后端） |
| M5 | 数据库与缓存（PostgreSQL + Redis） | ✅ 已完成（便携版 + 15 测试通过） |
| M6 | FastAPI 后端 | ✅ 已完成（端到端冒烟通过） |
| M7 | Vue3 前端 | ✅ 代码完成 + 构建通过 |
| M8 | 测试完善 + Docker 部署 | ✅ Docker 交付物已写；全链路冒烟通过（GA 完成后自动切换最优模型） |
| M9 | 文档与答辩支持 | ✅ 答辩要点已写（指标随 GA 完成后回填） |

## 数据集

- 本机已解压路径：`C:\Users\Administrator\Desktop\KolektorSDD\KolektorSDD`（399 张：52 缺陷 / 347 正常）
- M1 已整理到 `datasets/kolektor_sdd/` 并转换为 YOLO 训练格式

## 文档

- 规格说明书（设计蓝图 + AI 使用记录）：[docs/SPEC.md](docs/SPEC.md)
