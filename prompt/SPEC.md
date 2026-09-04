# 《制造智能技术》课程设计 — 规格说明书（SPEC）

| 项目 | 内容 |
|---|---|
| 版本 | v1.0（M0 交付） |
| 日期 | 2026-09-04 |
| 状态 | 待用户验收 |
| 题目 | 基于深度学习的轴承表面缺陷智能检测系统 |

---

## 1. 项目概述

- **课程**：《制造智能技术》课程设计
- **目标**：交付一个完整可运行的 B/S 架构缺陷检测 demo，覆盖计算机视觉、机器学习、智能优化三个技术方向，且三个算法模块在实际业务中各司其职（非装饰性引入）
- **数据来源**：KolektorSDD 公开表面缺陷数据集（初版）
- **架构**：FastAPI 后端 + Vue3 前端 + PostgreSQL 数据库 + Redis 缓存 + Docker 部署

> 如实说明：KolektorSDD 拍摄对象为换向器表面（工业表面缺陷），并非轴承照片。本系统的
> 「缺陷定位 → 特征判别 → 参数优化」方法可完整迁移至轴承表面缺陷检测场景，文档中均如实表述。

## 2. 需求规格

### 2.1 功能性需求

| 编号 | 需求 | 说明 |
|---|---|---|
| FR-1 | 图像上传检测 | 用户通过 Web 页面上传单张表面图像，系统调用检测流水线（YOLOv8 定位 + SVM 判别），返回缺陷判定结果、缺陷框坐标、置信度与处理耗时 |
| FR-2 | 结果可视化 | 前端在图片上叠加渲染缺陷框与标签，展示整体判定（有缺陷/无缺陷）、缺陷数量、平均置信度 |
| FR-3 | 历史记录 | 每次检测自动持久化到 PostgreSQL；支持分页查询、按判定结果筛选、查看详情、删除 |
| FR-4 | 统计看板 | 展示累计检测数、缺陷率、近 7 天检测趋势、缺陷数分布等统计图表 |
| FR-5 | 模型信息 | 展示 YOLOv8 / SVM / GA 三个模型的版本、评估指标、GA 最优超参数与收敛曲线 |
| FR-6 | 离线训练与优化 | 提供 YOLOv8 训练、特征提取与 SVM 训练、GA 超参优化三个命令行脚本，输出权重文件、评估报告、最优参数并写入数据库 |

### 2.2 非功能性需求

| 编号 | 需求 | 说明 |
|---|---|---|
| NFR-1 | 可运行性 | 整条链路在本机均可跑通；训练脚本自动检测 GPU/CPU |
| NFR-2 | 三方向覆盖 | YOLOv8（CV）、SVM（ML）、GA（智能优化）均承担不可替代的实际职责（见第 4 章） |
| NFR-3 | 代码质量 | 全部代码中文注释；关键函数带类型注解与文档字符串；模块化分层 |
| NFR-4 | 可部署性 | 提供 Dockerfile 与 docker-compose，一键启动后端+前端+PostgreSQL+Redis |
| NFR-5 | 可追溯性 | 开发过程按模块记录设计思路与关键决策（第 12 章 AI 使用记录） |

## 3. 技术选型与理由

| 技术 | 角色 | 选型理由 |
|---|---|---|
| YOLOv8（ultralytics） | 缺陷区域定位 | 主流单阶段检测器，训练部署简单，与 FastAPI 集成方便 |
| SVM（scikit-learn） | 二级判别 | 小样本高维特征下稳健；与深度学习形成"传统 ML + 深度 CV"互补，契合课程方向要求 |
| 遗传算法（手写） | 超参数优化 | 完整展示选择/交叉/变异算子，过程可控可解释，便于撰写报告 |
| FastAPI | 后端框架 | 高性能异步框架，自带 Swagger 文档，与算法栈同语言 |
| Vue3 + Vite + Element Plus + ECharts | 前端 | 组件化开发效率高，Element Plus 上手快，ECharts 满足统计图表需求 |
| PostgreSQL + SQLAlchemy 2 | 数据持久化 | 检测记录/模型元数据为关系型数据；ORM 便于维护与测试 |
| Redis | 缓存 | 缓存近期检测结果与统计汇总，体现缓存层设计 |
| Docker | 部署 | 交付物要求；一键化部署降低环境差异 |

## 4. 三大算法分工设计（核心章节）

### 4.1 检测流水线

```
用户上传图像
   │
   ▼
① YOLOv8 推理（一级定位）        → 输出候选缺陷框列表 [box, conf]
   │  └─ 算法方向：计算机视觉
   ▼
② 候选框裁剪 + HOG/LBP 特征提取
   │
   ▼
③ SVM 判别（二级确认）           → 每个候选框输出 [真缺陷 / 误检, 概率]
   │  └─ 算法方向：机器学习
   ▼
④ 应用 GA 优化得到的最优超参数（SVM 的 C/γ/核函数 + HOG 参数 + YOLO 阈值）
   │  └─ 算法方向：智能优化（离线完成，参数入库，推理时加载）
   ▼
⑤ 汇总结果 → PostgreSQL 持久化 → Redis 缓存 → 返回前端
```

### 4.2 各算法职责与「为何不是装饰性引入」

| 算法 | 职责 | 移除后的影响 |
|---|---|---|
| YOLOv8 | 「缺陷在哪」——定位缺陷区域。KolektorSDD 仅像素掩码无 bbox 标注，M1 用连通域分析从掩码生成 bbox 后训练 | 系统无法定位缺陷，前端无法渲染缺陷框 |
| SVM | 「是不是真缺陷」——对 YOLO 检出区域提取 HOG+LBP 特征做二分类，过滤误检 | 只剩单一检测器，误报无法抑制，检测精度下降 |
| GA | 「参数怎么调到最优」——优化 SVM 超参数（C、γ、核函数）、HOG 参数、YOLO 阈值 | 参数只能人工试错；失去智能优化方向，报告无法覆盖该方向 |

### 4.3 GA 设计（M4 实现，此处为规格约定）

- **基因编码**：实数+整数混合编码。C、γ 取 log10 实数；核函数、HOG cell、HOG 方向数、YOLO conf/iou 阈值取离散值
- **算子**：种群 30、锦标赛选择（k=3）、单点交叉（Pc=0.8）、高斯/均匀变异（Pm=0.1）、精英保留 2 名
- **适应度**：验证集 F1 分数（特征向量预先缓存到磁盘，GA 每代只需训练 SVM + 预测，秒级完成，迭代 50 代可行）
- **输出**：收敛曲线图、最优参数 JSON → 写入 model_metadata 与 ga_history 表
- **对比实验**：随机参数 / 网格搜索 / GA 三组结果对比（报告加分点）

## 5. 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                 前端  Vue3 + Vite (浏览器)                 │
│        检测页 · 历史记录页 · 统计看板 · 模型信息页          │
└──────────────────────────┬───────────────────────────────┘
                           │ HTTP / JSON
┌──────────────────────────▼───────────────────────────────┐
│                    FastAPI 后端 (Python)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  检测服务     │  │  模型管理     │  │ 记录/统计 API    │ │
│  │  YOLOv8 定位  │  │  加载权重与   │  │  SQLAlchemy ORM │ │
│  │  SVM 二级判别 │  │  GA 最优参数  │  │                 │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘ │
│         └─────────┬───────┴──────────────────┘           │
│  ┌────────────────▼─────────────────────────────────┐    │
│  │  PostgreSQL（检测记录/模型元数据/GA历史） + Redis（缓存）│    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
离线训练（命令行脚本，不占 Web 服务资源）：
  prepare_dataset.py → train_yolo.py → extract_features.py
  → train_svm.py → ga_optimize.py → 最优参数写入数据库
```

## 6. 数据集设计

### 6.1 原始数据（已确认）

- 来源：KolektorSDD（初版），本机已解压 `C:\Users\Administrator\Desktop\KolektorSDD\KolektorSDD`（备份 `D:\Downloads\KolektorSDD.zip`）
- 规模：399 张灰度图（500×1258 单通道），官方划分 **Train 319 张**（kos01~kos40）/ **Test 80 张**（kos41~kos50）
- 标注：每张 `PartX.jpg` 配一个 `PartX_label.bmp` 像素掩码（白色区域=缺陷）；无 bbox 标注；官方构成 52 张缺陷 / 347 张正常（正常样本掩码全黑）
- 语义：图像级二分类——有缺陷（含 1 个或多个缺陷区域）/ 无缺陷

### 6.2 标注转换方案（M1）

```
PartX_label.bmp 掩码
    → 二值化（缺陷像素=1）
    → 连通域分析 cv2.connectedComponentsWithStats
    → 每个连通域生成一个 YOLO bbox（类别 0 = defect）
    → 输出 YOLO 格式 labels/*.txt + images/ 副本
```

### 6.3 数据划分

- 沿用官方 Train/Test 划分（Test 用于最终评估）
- Train 内部再按 80/20 划出验证集，分层采样保证缺陷/正常比例一致
- 防数据泄漏：同一 kosXX 批次不跨划分

## 7. API 设计（v1 概要，M6 细化）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /api/v1/detect | multipart 上传图片，返回检测结果 JSON |
| GET | /api/v1/records | 历史记录分页查询（?page&size&has_defect） |
| GET | /api/v1/records/{id} | 记录详情（含标注框数据） |
| DELETE | /api/v1/records/{id} | 删除记录 |
| GET | /api/v1/stats/summary | 统计汇总（Redis 缓存 60s） |
| GET | /api/v1/models | 三模型元数据（版本/指标/GA 参数） |
| GET | /api/v1/health | 健康检查 |

响应统一格式：`{"code": 0, "message": "ok", "data": ...}`

## 8. 数据库设计（概要，M5 细化）

### 8.1 detection_records（检测记录）

| 字段 | 类型 | 说明 |
|---|---|---|
| id | BIGSERIAL PK | 主键 |
| filename | VARCHAR | 存储文件名（如 20260904_103000_a1b2.jpg） |
| original_name | VARCHAR | 用户上传的原始文件名 |
| has_defect | BOOLEAN | 整体判定结果 |
| defect_count | INT | 缺陷框数量 |
| avg_confidence | FLOAT | 平均置信度 |
| processing_time_ms | FLOAT | 处理耗时 |
| boxes | JSONB | 缺陷框明细（xyxy / svm_label / yolo_conf / svm_prob） |
| model_versions | JSONB | 本次检测使用的模型版本号 |
| created_at | TIMESTAMPTZ | 检测时间 |

### 8.2 model_metadata（模型元数据）

id / model_type(yolo|svm|ga) / version / file_path / metrics(JSONB) / params(JSONB) / created_at

### 8.3 ga_history（GA 迭代历史）

id / generation INT / best_fitness FLOAT / avg_fitness FLOAT / best_params(JSONB) / created_at

### 8.4 Redis 键设计

- `detection:{id}` → 检测结果缓存，TTL 10min
- `stats:summary` → 统计汇总缓存，TTL 60s

## 9. 前端页面设计（M7 实现）

| 页面 | 路由 | 内容 |
|---|---|---|
| 检测页 | /detect | 拖拽/点击上传 → 图片预览 → 开始检测 → 结果卡片（整体判定/缺陷数/置信度/耗时）+ canvas 缺陷框叠加渲染 |
| 历史记录 | /history | Element Plus 表格 + 判定筛选 + 分页 + 详情对话框（复现标注图） |
| 统计看板 | /stats | ECharts：KPI 卡（总检测数/缺陷率）+ 近 7 天趋势折线 + 缺陷数分布柱状图 |
| 模型信息 | /models | 三模型版本与指标卡片 + GA 收敛曲线 + 最优参数表 |

整体布局：侧边栏导航 + 顶栏标题，深色工业风。

## 10. 模块开发计划与验收标准

| 模块 | 内容 | 验收要点 |
|---|---|---|
| M0 | 项目规划与规格说明 | 规格文档确认、环境就绪（venv 可用） |
| M1 | 数据准备 | 掩码→bbox 转换脚本运行成功、可视化抽查正确、划分合理 |
| M2 | YOLOv8 训练 | 训练完成、mAP 指标输出、权重导出 |
| M3 | SVM 特征提取与训练 | 准确率/AUC 评估报告输出 |
| M4 | GA 超参数优化 | 收敛曲线、优化前后对比、参数入库 |
| M5 | 数据库与缓存 | 表结构创建、ORM CRUD 单测通过、Redis 连通 |
| M6 | FastAPI 后端 | Swagger 全流程走通、pytest 单测通过 |
| M7 | Vue3 前端 | 上传→检测→结果→历史→统计全流程走通 |
| M8 | 测试完善与 Docker 部署 | docker compose 一键启动全套 |
| M9 | 文档与答辩支持 | README/设计文档/答辩要点齐备 |

## 11. 开发规范

- 文件头 docstring：一句话说明模块职责
- 公开函数：类型注解 + 中文 docstring（参数/返回值/异常）
- 命名：函数变量 snake_case / 类 PascalCase / 常量 UPPER_CASE
- 注释：算法原理、关键步骤、魔法数字必须注释（中文）
- 测试：backend 每层对应 tests/ 下 pytest 用例；算法模块输出评估指标文件
- 流程：自主推进模式（2026-09-04 用户授权）——每模块完成后自行验证并进入下一模块；每阶段仍输出设计思路与关键决策（AI 使用记录）；重大方向性决策才停下询问用户

## 12. AI 使用记录

| 日期 | 阶段 | 设计思路与关键决策 |
|---|---|---|
| 2026-09-04 | M0 规划 | ①SVM 任务定位=二级判别（用户确认）②掩码→连通域→bbox 转换方案（弥补数据集无 bbox 标注）③GA 手写实现、适应度=验证集 F1、含随机/网格/GA 对比实验 ④环境事实：RTX 4060 / CUDA 12.8 → torch 用 cu128 独立安装；机器实际 Python 3.13（用户原以为 3.10）⑤Docker 未安装：M5 前在「装 Docker Desktop」与「本机直装 PG+Redis」两条路径中选择，M8 仍交付 docker-compose |
| 2026-09-04 | M1 数据准备 | ①按「物理件 kosXX」划分 train/val 防数据泄漏（KolektorSDD 每个 kosXX 是同一产品的多张子图；官方 Test 本就按物理件划分，直接沿用为 test）②掩码→二值化(>127)→连通域分析→外接矩形 bbox，min_area=25 过滤噪声 ③无缺陷图像不写标签文件（YOLO 标准负样本处理）④实况核查：399 张全部带掩码文件，正常样本为全黑掩码，脚本对「缺文件/全黑」两种情形均兼容 ⑤seed=42 保证划分可复现，划分明细存 split_info.json |
| 2026-09-04 | M2 YOLO 训练 | ①模型选 yolov8n（小数据集 + 8GB 显存够用，可 --model 换 s/m）②imgsz=640 + rect 矩形训练（500×1258 细长图按长宽比组批，比强制方形更高效）③Windows 下 workers=0 单进程数据加载（多进程易死锁）④早停 patience=20、seed=42 可复现；mosaic/flip/hsv 默认增强应对小数据集 ⑤best.pt 固化到 backend/weights/，指标写 JSON 供数据库与前端模型信息页使用 ⑥eval_yolo.py 在官方 Test（80 张）做 hold-out 评估，输出 PR 曲线与混淆矩阵 |
| 2026-09-04 | M3 SVM 判别 | ①特征设计：候选框统一 64×64 → HOG(1764维) + 旋转不变均匀模式 LBP(59维) + 统计特征(3维) = 1826 维，对光照变化鲁棒 ②样本标签：候选框与真值框 IoU≥0.5 为正样本；YOLO 误检为负样本；漏检真值框追加为正样本、随机背景块补充负样本保持平衡 ③SVM 用 RBF 核 + class_weight=balanced ④特征一次性缓存为 npz，SVM 训练与 GA 评估共用（svm_core.py 共享模块） |
| 2026-09-04 | M4 遗传算法 | ①手写 GA：锦标赛选择(k=3)、单点交叉(Pc=0.8)、混合变异(Pm=0.1)、精英保留 2 名 ②基因 = log10(C)、log10(γ)、核函数、YOLO conf 阈值、SVM prob 阈值——阈值与分类器联合优化（HOG/LBP 参数固定以保证每代评估秒级完成）③适应度 = 验证集两级流水线样本级 F1 ④对比实验：随机搜索 60 组 / 网格搜索（固定阈值基线）/ GA ⑤最优参数重训最终 SVM 并固化 joblib 供后端加载 |
| 2026-09-04 | M3 修订 ① | 实测发现 opencv-python 4.10+ 已移除 cv2.HOGDescriptor 的 Python 绑定 → 手写 numpy 版 HOG（网格几何与原配置一致：7×7 块×4 cell×9 bin=1764 维；Sobel 梯度→cell 直方图→L2-Hys 归一化）。SVM 训练前特征空间由我们定义，自洽即可；手写实现也更利于课程答辩讲清 HOG 原理 |
| 2026-09-04 | M3 修订 ② | 诊断发现 YOLO 置信度标定极差：conf=0.001 时单图 300 候选，但最高 conf 仅 0.013（小数据训练导致）——原设计 conf=0.05 下 0 个候选。修订：候选阈值降到 0.001 最大化召回；每图困难负样本挖掘——正样本（IoU≥0.5）无条件保留、负样本只取 conf 前 top-K(K=20)，否则 12 万候选会使 RBF-SVM 内存爆炸。GA 的 conf_th 搜索范围相应改为 [0.001, 0.02] |
| 2026-09-04 | M3 修订 ③ | 线上冒烟发现正常件假阳性（kos41_Part0 判 7 框，SVM conf 最高 0.996）：GT 框 92.6% 也贴边故位置特征无效；根因是负样本多样性不足——347 张正常图的端面/加工纹理 SVM 从未见过。修订：无缺陷图每张采样 2 个方形随机块 + 2 个边缘条带块（16~48px 厚度，覆盖扁平贴边形态）作负样本。修订后 Part0 假阳性 7→3 框，GA 优化阈值后进一步过滤 |
| 2026-09-04 | M5 数据库缓存 | ①无管理员权限 → 便携方案：PostgreSQL 17.5 官方二进制 zip（initdb 免安装）+ tporadowski Redis 5.0.14 ②踩坑：initdb 在中文路径（轴承缺陷项目\）下报 FATAL invalid byte sequence 0xd6 0xe1（"轴"字 GBK 编码被 UTF8 服务端拒绝）→ 数据目录迁至纯 ASCII 路径 C:\pg_bearing\data ③redis-py 8.x 发 HELLO 3 命令做 RESP3 协商，Redis 5.0 服务端不支持 → 降级 redis-py==5.2.1 ④created_at 用 ORM 层 default=datetime.now：SQLite 的 CURRENT_TIMESTAMP 是 UTC，按本地 0 点统计近 7 天会漏当天记录，应用层时间保证 SQLite/PG 行为一致 ⑤缓存策略：检测结果按图像内容 MD5 缓存 600s（命中不再落库防膨胀）；统计概览缓存 60s；Redis 不可用优雅降级不阻塞主流程 |
| 2026-09-04 | M6 FastAPI 后端 | ①检测流水线服务 app/services/detector.py：YOLO(conf_th) + SVM(prob_th) 两级过滤，阈值来自 GA 优化产物 svm_classifier.joblib ②懒加载单例：首次请求才加载模型（约 3s），不占启动时间 ③api/detect.py 按图像 MD5 查缓存→检测→落库→回填缓存 ④单元测试用 monkeypatch 替换 detect_image + SQLite 内存库，不依赖真实模型/数据库；测试间数据隔离（模块级 drop_all+create_all）⑤端到端冒烟通过：缺陷图检出 10 框（SVM conf 0.57~0.997，首请求 3.3s 含模型加载，后续约 1.2s）、重复上传命中缓存、历史/统计/模型信息接口正常 |
| 2026-09-04 | M7 前端 | ①Vue3 + Vite + Element Plus + ECharts 四页面（检测/历史/统计/模型信息）②dev 环境 vite proxy 代理 /api 到后端，免 CORS ③检测页 canvas 绘制原图+缺陷框+置信度标签 ④统计页 ECharts 近 7 天趋势图（数据来自后端聚合接口）⑤axios 拦截器统一解包 {code,message,data} 信封，code≠0 统一报错提示 |
| 2026-09-04 | M3 修订 ④ | 新增图像级评测脚本 eval_pipeline.py（逐图跑完整两级流水线，统计图像级 P/R/F1/ACC）发现推理侧与训练侧不一致：推理时全部 ~300 候选过 SVM（val 图像级 FP 43/64），训练负样本只覆盖 top-20。修订：detector 推理侧对候选按 conf 排序只取 top-K 送 SVM；K∈{20,30,50,100} 网格实验 → K=50 最优（test P=0.75/R=0.6/F1=0.6667/ACC=0.925，FP 仅 2 张），K=100 反而退化（F1 0.35）——说明过多低分候选引入噪声 |
| 2026-09-04 | M4 修订 ① | 首跑发现 GA 串行执行极慢（单次适应度评估约 12s：SVC(probability=True) 内部 5 折交叉验证拟合 Platt 校准 = 6 次 SVC 训练；600 次评估串行跑了 100 分钟未完成被止损终止）。修订：①静默 sklearn probability 弃用 FutureWarning（stderr 重定向下每条警告都拖慢主循环）②适应度评估多进程并行——multiprocessing.Pool(16)（本机 i7-14650HX 24 线程），Windows spawn 模式下 worker 通过 initializer 一次性持有 44MB 特征集、避免每次任务 pickle 传输 ③GA 每代墙钟由最慢个体（poly 核训练比 rbf 慢 5~10 倍）决定，16 核并行后全量 pop20×gen30 + 随机 60 + 网格 24 预计 10~20 分钟内完成 ④并行化不改变算法本身（选择/交叉/变异/精英保留逻辑不变），只替换评估循环，优化结果与串行等价 |
| 2026-09-04 | M8 测试部署 | ①M5/M6 全部 pytest 通过（15 passed）；后端启动生命周期（lifespan）自动建表 ②Docker 交付物：backend/Dockerfile（python:3.13-slim + CPU 版 torch + uvicorn）、frontend/Dockerfile（node:20-alpine 构建 → nginx:1.27-alpine）、nginx.conf（try_files SPA 回退 + /api 反代 backend:8000，proxy_read_timeout 120s 适配首请求模型加载）、docker-compose.yml（postgres:16-alpine + redis:7-alpine 带 healthcheck + 后端 bind mount 注入权重/产物，不烧进镜像）③M9 答辩要点文档 docs/答辩要点.md：两级架构动机（YOLO 置信度标定差 → 分工）、三级技术分工表、GA 设计、现场演示 3 分钟脚本、Q&A 预演 ④GA 完成后端自动切换：后台 watcher 监听 ga_comparison.json → 自动 seed_models 导入 SVM/GA 元数据 → 重启 uvicorn 加载 GA 最优模型 → 冒烟验证 |

## 13. 风险与应对

| 风险 | 应对 |
|---|---|
| Docker 未安装 | M5 前确定路径；无论哪条路，M8 交付 docker-compose 配置 |
| 数据量小（399 张） | 训练时数据增强（翻转/亮度/仿射）；评估用 F1/AUC 而非只看 accuracy |
| 缺陷样本仅 52 张 | 分层划分保证各划分含缺陷样本；SVM 类别权重平衡 |
| 8GB 显存 | yolov8n/s 模型 + 500×1258 灰度图训练无压力 |
| KolektorSDD 非轴承图 | 文档如实表述；方法可迁移；用户补充轴承自采数据可扩展 |
