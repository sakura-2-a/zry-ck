"""
M5 种子脚本：把离线训练产出的指标导入数据库（前端"模型信息/统计"页数据源）。

导入内容：
  - model_metadata：YOLO 训练/Test 评估指标、SVM 验证指标、GA 最优参数
  - ga_history：GA 逐代历史（幂等：每次先清空再写入，保证与 artifacts 一致）

前置条件：PostgreSQL 已启动（便携版启动方式见 README M5 章节）。

用法（项目根目录，venv 已激活）：
    python backend/scripts/seed_models.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import delete, select

from app.core.database import SessionLocal, init_db
from app.models import GAHistory, ModelMetadata

ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

# 待导入的模型元数据：(模型名, 版本, 指标 json 文件名, 时间字段名)
MODEL_SOURCES = [
    ("YOLOv8n", "v1", "yolo_train_metrics.json", "trained_at"),
    ("YOLOv8n", "test-eval", "yolo_test_metrics.json", "evaluated_at"),
    ("SVM", "ga-best", "svm_val_metrics.json", "optimized_at"),
    ("GA", "best-params", "ga_best_params.json", None),
]


def load_json(name: str) -> dict | None:
    """读取 artifacts 下的 JSON 文件；不存在返回 None。"""
    path = ARTIFACTS_DIR / name
    if not path.exists():
        print(f"[跳过] 未找到 {path.name}（对应离线步骤尚未运行）")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def seed_model_metadata(db) -> int:
    """导入模型元数据（幂等：按 model_name+version 更新或插入）。"""
    count = 0
    for model_name, version, json_name, time_key in MODEL_SOURCES:
        data = load_json(json_name)
        if data is None:
            continue
        # 时间字段单独入库，其余全部进 metrics_json
        trained_at = None
        if time_key and data.get(time_key):
            try:
                trained_at = datetime.fromisoformat(data[time_key])
            except ValueError:
                trained_at = None
        metrics = {k: v for k, v in data.items() if k != time_key}

        row = db.execute(select(ModelMetadata).where(
            ModelMetadata.model_name == model_name,
            ModelMetadata.version == version,
        )).scalar_one_or_none()
        if row is None:
            db.add(ModelMetadata(model_name=model_name, version=version,
                                 metrics_json=metrics, trained_at=trained_at))
        else:
            row.metrics_json = metrics
            row.trained_at = trained_at
        count += 1
    db.commit()
    print(f"[模型元数据] 导入/更新 {count} 条")
    return count


def seed_ga_history(db) -> int:
    """导入 GA 逐代历史（先清空再写入）。"""
    data = load_json("ga_history.json")
    if data is None:
        return 0
    db.execute(delete(GAHistory))
    for h in data:
        db.add(GAHistory(generation=h["generation"], best_fitness=h["best"],
                         avg_fitness=h["avg"], params_json={}))
    db.commit()
    print(f"[GA 历史] 导入 {len(data)} 代")
    return len(data)


def main() -> None:
    """种子导入入口。"""
    init_db()
    db = SessionLocal()
    try:
        seed_model_metadata(db)
        seed_ga_history(db)
    finally:
        db.close()
    print("种子数据导入完成")


if __name__ == "__main__":
    main()
