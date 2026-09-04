"""
M4 遗传算法超参数优化脚本：手写 GA 优化「YOLO 阈值 + SVM 超参数」。

优化变量（基因，实数+整数混合编码）：
  0. log10(C)        连续  [-3, 3]      SVM 惩罚系数
  1. log10(gamma)    连续  [-4, 1]      RBF 核宽度（linear 核时被忽略）
  2. kernel          离散  {rbf, linear, poly}
  3. conf_th         连续  [0.001, 0.02]  YOLO 置信度阈值（训练后模型 conf 普遍 <0.02）
  4. prob_th         连续  [0.3, 0.9]   SVM 概率阈值

适应度 = 验证集「两级流水线」样本级 F1（特征已缓存，每代评估秒级完成）。

算法要素（手写实现，便于报告讲解）：
  种群初始化：边界内均匀随机
  选择：锦标赛选择（k=3）
  交叉：单点交叉（Pc=0.8）
  变异：连续基因高斯变异（σ=0.15×范围），离散基因（kernel）均匀重采样（Pm=0.1）
  精英保留：每代最优 2 个个体直接进入下一代
  终止：达到最大代数

输出（backend/artifacts/）：
  ga_best_params.json       最优参数与适应度
  ga_convergence.png        收敛曲线（逐代最优/平均）
  ga_history.json           逐代历史
  ga_comparison.png/json    随机搜索 / 网格搜索 / GA 对比（--mode all）
  svm_classifier.joblib     用最优参数重训的最终模型（覆盖 M3 基线）

用法（项目根目录，venv 已激活）：
    python backend/scripts/ga_optimize.py                     # 运行 GA
    python backend/scripts/ga_optimize.py --generations 50 --pop-size 30
    python backend/scripts/ga_optimize.py --mode all          # 含对比实验
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # 无界面后端，直接保存图片文件
import matplotlib.pyplot as plt
import numpy as np

# sklearn 1.9 起 probability 参数弃用（每次训练打一条 FutureWarning，stderr 重定向下
# 严重拖慢 GA 主循环）——本项目用的 SVC(probability=True) 行为不变，静默处理
warnings.filterwarnings("ignore", category=FutureWarning)

# 同目录共享模块（特征加载 / SVM 训练 / 流水线评估 + UTF-8 控制台）
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import io_utils
import svm_core

# 项目根目录（backend/scripts/ 向上两级）
PROJECT_ROOT = SCRIPT_DIR.parents[1]
BACKEND_DIR = SCRIPT_DIR.parent
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

# Windows 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 基因边界（与模块 docstring 中的说明一一对应）
GENE_BOUNDS = np.array([
    [-3.0, 3.0],     # log10(C)
    [-4.0, 1.0],     # log10(gamma)
    [0.0, 2.0],      # kernel 索引 {0: rbf, 1: linear, 2: poly}
    [0.001, 0.02],   # YOLO conf 阈值（训练后模型 conf 普遍 < 0.02，见 diag_conf.py）
    [0.3, 0.9],      # SVM prob 阈值
])
KERNELS = ["rbf", "linear", "poly"]
KERNEL_IDX = 2        # kernel 基因在向量中的下标
PC = 0.8              # 交叉概率
PM = 0.1              # 变异概率
ELITE = 2             # 精英保留数量
TOURNAMENT_K = 3      # 锦标赛规模


# ---------------------------------------------------------------------------
# 编码与适应度
# ---------------------------------------------------------------------------
def decode(genome: np.ndarray) -> dict[str, float | str]:
    """把基因向量解码为可直接训练的超参数字典。

    参数:
        genome: 长度 5 的实数基因向量（kernel 位取整使用）

    返回:
        {"C": float, "gamma": float, "kernel": str, "conf_th": float, "prob_th": float}
    """
    return {
        "C": float(10 ** genome[0]),
        "gamma": float(10 ** genome[1]),
        "kernel": KERNELS[int(round(float(genome[2])))],
        "conf_th": float(genome[3]),
        "prob_th": float(genome[4]),
    }


def fitness(genome: np.ndarray, train_set: dict, val_set: dict) -> float:
    """计算个体适应度：按 conf 阈值过滤训练样本 → 训练 SVM → 验证集流水线 F1。

    参数:
        genome: 基因向量
        train_set, val_set: svm_core.load_feature_set 返回的缓存特征集

    返回:
        验证集 F1（0~1），不可行个体返回 0
    """
    params = decode(genome)
    # 1. 按 YOLO 置信度阈值过滤训练候选框（阈值与分类器联合优化）
    keep = train_set["conf"] >= params["conf_th"]
    if keep.sum() < 10 or (train_set["y"][keep] == 1).sum() < 3:
        return 0.0  # 过滤后样本过少或正样本不足，判为不可行

    scaler, svm = svm_core.train_svm(
        train_set["X"][keep], train_set["y"][keep],
        C=params["C"], gamma=params["gamma"], kernel=params["kernel"],
    )
    return svm_core.evaluate_pipeline(
        svm, scaler, val_set["X"], val_set["y"], val_set["conf"],
        params["conf_th"], params["prob_th"],
    )["f1"]


def random_genome(rng: np.random.Generator) -> np.ndarray:
    """在边界内均匀随机初始化一个基因向量。

    参数:
        rng: 随机数生成器

    返回:
        长度 5 的基因向量
    """
    return rng.uniform(GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1])


# ---------------------------------------------------------------------------
# 多进程并行评估（Windows spawn 模式下 worker 通过 initializer 持有特征集，
# 避免每次任务 pickle 传输 44MB 特征矩阵）
# ---------------------------------------------------------------------------
_WORKER_TRAIN: dict | None = None
_WORKER_VAL: dict | None = None
_MAIN_TRAIN: dict | None = None
_MAIN_VAL: dict | None = None


def _init_worker(train_set: dict, val_set: dict) -> None:
    """Pool worker 初始化：把特征集挂到进程级全局变量。"""
    global _WORKER_TRAIN, _WORKER_VAL
    _WORKER_TRAIN, _WORKER_VAL = train_set, val_set


def _fitness_worker(genome: np.ndarray) -> float:
    """worker 端适应度包装：从进程级全局取特征集。"""
    return fitness(genome, _WORKER_TRAIN, _WORKER_VAL)


def _map_fitness(pool: Pool | None, genomes: list[np.ndarray]) -> np.ndarray:
    """评估一组基因的适应度：有 Pool 时并行，无 Pool 时串行。

    参数:
        pool: multiprocessing.Pool（None 表示串行）
        genomes: 待评估基因列表

    返回:
        适应度数组（顺序与输入一致）
    """
    if pool is None:
        return np.array([fitness(g, _MAIN_TRAIN, _MAIN_VAL) for g in genomes])
    return np.array(pool.map(_fitness_worker, genomes))


# ---------------------------------------------------------------------------
# 遗传算子（选择 / 交叉 / 变异）
# ---------------------------------------------------------------------------
def tournament_select(
    pop: np.ndarray, fits: np.ndarray, rng: np.random.Generator, k: int = TOURNAMENT_K
) -> np.ndarray:
    """锦标赛选择：随机抽 k 个个体，返回其中适应度最高的基因（返回副本）。

    参数:
        pop: 种群矩阵 (n, 5)
        fits: 适应度向量 (n,)
        rng: 随机数生成器
        k: 锦标赛规模

    返回:
        选中个体的基因副本
    """
    idx = rng.integers(0, len(pop), size=k)
    best = idx[int(np.argmax(fits[idx]))]
    return pop[best].copy()


def single_point_crossover(
    parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator, pc: float = PC
) -> np.ndarray:
    """单点交叉：以概率 pc 随机选一个断点，交换两亲本断点之后的基因段。

    参数:
        parent1, parent2: 亲本基因
        rng: 随机数生成器
        pc: 交叉概率

    返回:
        子代基因
    """
    if rng.random() > pc:
        return parent1.copy()  # 未发生交叉，直接继承亲本 1
    point = int(rng.integers(1, len(parent1)))  # 断点 ∈ [1, n-1]
    return np.concatenate([parent1[:point], parent2[point:]])


def mutate(
    genome: np.ndarray, rng: np.random.Generator, pm: float = PM
) -> np.ndarray:
    """混合变异：连续基因加高斯噪声，离散基因（kernel）均匀重采样。

    参数:
        genome: 待变异基因
        rng: 随机数生成器
        pm: 单基因变异概率

    返回:
        变异后的基因（已截断回边界）
    """
    child = genome.copy()
    for i in range(len(child)):
        if rng.random() > pm:
            continue
        if i == KERNEL_IDX:
            child[i] = rng.uniform(*GENE_BOUNDS[i])        # 离散基因：重新随机
        else:
            sigma = 0.15 * (GENE_BOUNDS[i, 1] - GENE_BOUNDS[i, 0])
            child[i] += rng.normal(0.0, sigma)             # 连续基因：高斯扰动
        child[i] = float(np.clip(child[i], *GENE_BOUNDS[i]))  # 截断回边界
    return child


# ---------------------------------------------------------------------------
# 遗传算法主循环
# ---------------------------------------------------------------------------
def run_ga(
    train_set: dict, val_set: dict, pop_size: int = 20, generations: int = 30,
    seed: int = 42, pool: Pool | None = None,
) -> dict:
    """执行遗传算法主循环。

    参数:
        train_set, val_set: 缓存特征集
        pop_size: 种群规模
        generations: 最大代数
        seed: 随机种子
        pool: 多进程池（None 为串行）

    返回:
        {"history": [{"generation", "best", "avg"}, ...],
         "best_genome": np.ndarray, "best_fitness": float}
    """
    rng = np.random.default_rng(seed)
    pop = np.array([random_genome(rng) for _ in range(pop_size)])
    fits = _map_fitness(pool, [g for g in pop])

    history: list[dict] = []
    for gen in range(generations):
        history.append({
            "generation": gen + 1,
            "best": round(float(fits.max()), 4),
            "avg": round(float(fits.mean()), 4),
        })
        if gen == generations - 1:
            break

        # 生成下一代：精英保留 + 选择/交叉/变异
        new_pop = [pop[i].copy() for i in np.argsort(fits)[-ELITE:]]  # 最优 ELITE 个直接保留
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fits, rng)
            p2 = tournament_select(pop, fits, rng)
            new_pop.append(mutate(single_point_crossover(p1, p2, rng), rng))
        pop = np.array(new_pop)
        fits = _map_fitness(pool, [g for g in pop])
        print(f"    第 {gen + 2} 代: 最优 F1 = {fits.max():.4f}, 平均 = {fits.mean():.4f}")

    best_idx = int(np.argmax(fits))
    return {
        "history": history,
        "best_genome": pop[best_idx],
        "best_fitness": float(fits[best_idx]),
    }


# ---------------------------------------------------------------------------
# 对比基线：随机搜索 / 网格搜索
# ---------------------------------------------------------------------------
def run_random_search(
    train_set: dict, val_set: dict, n_trials: int = 60, seed: int = 42,
    pool: Pool | None = None,
) -> dict:
    """随机搜索基线：随机采样 n_trials 组参数取最优。

    参数:
        train_set, val_set: 缓存特征集
        n_trials: 采样组数（与 GA 评估次数同数量级，保证对比公平）
        seed: 随机种子
        pool: 多进程池（None 为串行）

    返回:
        {"best_genome", "best_fitness", "samples": [...]}
    """
    rng = np.random.default_rng(seed)
    genomes = [random_genome(rng) for _ in range(n_trials)]
    fits = _map_fitness(pool, genomes)
    best_idx = int(np.argmax(fits))
    samples = [
        {"params": decode(g), "fitness": round(float(f), 4)}
        for g, f in zip(genomes, fits)
    ]
    return {"best_genome": genomes[best_idx],
            "best_fitness": float(fits[best_idx]), "samples": samples}


def run_grid_search(train_set: dict, val_set: dict, pool: Pool | None = None) -> dict:
    """网格搜索基线：C/γ/核函数小网格 + 固定阈值（conf=0.01, prob=0.5）。

    代表"人工调参"的常规做法：只搜 SVM 参数，阈值凭经验固定。

    参数:
        train_set, val_set: 缓存特征集
        pool: 多进程池（None 为串行）

    返回:
        {"best_genome", "best_fitness", "samples": [...]}
    """
    genomes = []
    for C in (0.1, 1.0, 10.0, 100.0):
        for gamma in (0.001, 0.01, 0.1):
            for kernel in ("rbf", "linear"):
                genomes.append(np.array([np.log10(C), np.log10(gamma),
                                         float(KERNELS.index(kernel)), 0.01, 0.5]))
    fits = _map_fitness(pool, genomes)
    best_idx = int(np.argmax(fits))
    samples = [
        {"params": decode(g), "fitness": round(float(f), 4)}
        for g, f in zip(genomes, fits)
    ]
    return {"best_genome": genomes[best_idx],
            "best_fitness": float(fits[best_idx]), "samples": samples}


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------
def plot_convergence(history: list[dict], out_path: Path) -> None:
    """绘制 GA 逐代最优/平均适应度收敛曲线。

    参数:
        history: run_ga 返回的逐代历史
        out_path: 输出图片路径
    """
    gens = [h["generation"] for h in history]
    best = [h["best"] for h in history]
    avg = [h["avg"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best, "-o", color="#e74c3c", label="最优适应度")
    ax.plot(gens, avg, "--s", color="#3498db", label="平均适应度")
    ax.set_xlabel("代数")
    ax.set_ylabel("验证集 F1")
    ax.set_title("遗传算法收敛曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[GA] 收敛曲线已保存: {out_path}")


def plot_comparison(results: dict[str, dict], out_path: Path) -> None:
    """绘制三种优化方法的最优 F1 对比柱状图。

    参数:
        results: {"random": {"best_fitness", ...}, "grid": {...}, "ga": {...}}
        out_path: 输出图片路径
    """
    labels = {"ga": "遗传算法", "random": "随机搜索", "grid": "网格搜索"}
    names = [labels[k] for k in results]
    values = [results[k]["best_fitness"] for k in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=["#e74c3c", "#95a5a6", "#3498db"], width=0.5)
    ax.set_ylabel("验证集 F1")
    ax.set_title("超参数优化方法对比")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[对比] 对比图已保存: {out_path}")


def save_final_model(params: dict, train_set: dict, val_set: dict) -> None:
    """用最优参数在完整训练集上重训 SVM，固化为 joblib（后端直接加载）。

    参数:
        params: decode 得到的最优超参数字典
        train_set, val_set: 缓存特征集
    """
    keep = train_set["conf"] >= params["conf_th"]
    scaler, svm = svm_core.train_svm(
        train_set["X"][keep], train_set["y"][keep],
        C=params["C"], gamma=params["gamma"], kernel=params["kernel"],
    )
    bundle = {
        "scaler": scaler,
        "svm": svm,
        "feature_config": svm_core.FEATURE_CONFIG,
        "conf_th": params["conf_th"],
        "prob_th": params["prob_th"],
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACTS_DIR / "svm_classifier.joblib")

    # 同时存档最优模型的验证集指标（供数据库与前端模型信息页使用）
    pipe_metrics = svm_core.evaluate_pipeline(
        svm, scaler, val_set["X"], val_set["y"], val_set["conf"],
        params["conf_th"], params["prob_th"],
    )
    val_metrics = {
        "optimized_at": datetime.now().isoformat(timespec="seconds"),
        "params": {k: (v if isinstance(v, (int, float)) else v) for k, v in params.items()},
        "pipeline_metrics": pipe_metrics,
    }
    (ARTIFACTS_DIR / "svm_val_metrics.json").write_text(
        json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[GA] 最优模型已固化: {ARTIFACTS_DIR / 'svm_classifier.joblib'}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="遗传算法超参数优化")
    parser.add_argument("--mode", type=str, default="ga",
                        choices=("ga", "random", "grid", "all"),
                        help="运行模式：ga/random/grid/all(含对比)")
    parser.add_argument("--pop-size", type=int, default=20, help="种群规模")
    parser.add_argument("--generations", type=int, default=30, help="最大代数")
    parser.add_argument("--random-trials", type=int, default=60,
                        help="随机搜索采样组数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="并行评估进程数（1=串行；SVC 训练释放 GIL，多进程可近似线性加速）")
    parser.add_argument("--train-path", type=Path,
                        default=ARTIFACTS_DIR / "svm_features_train.npz",
                        help="训练特征缓存路径")
    parser.add_argument("--val-path", type=Path,
                        default=ARTIFACTS_DIR / "svm_features_val.npz",
                        help="验证特征缓存路径")
    return parser.parse_args()


def main() -> None:
    """GA 优化入口。"""
    io_utils.force_utf8_stdout()
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # 把 stdout/stderr 重定向到日志文件：后台运行时外层管道可能中途关闭，
    # 主进程 print 写已关闭管道会抛 BrokenPipeError 致死（曾致 GA 两次中断）。
    # 直写文件后脚本完全独立于启动终端/管道的生命周期。
    _log = open(ARTIFACTS_DIR / "ga_run.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _log
    sys.stderr = _log
    train_set = svm_core.load_feature_set(args.train_path)
    val_set = svm_core.load_feature_set(args.val_path)

    # 并行评估：主进程串行路径走全局特征集，worker 走 initializer 注入的副本
    global _MAIN_TRAIN, _MAIN_VAL
    _MAIN_TRAIN, _MAIN_VAL = train_set, val_set
    pool: Pool | None = None
    if args.n_jobs > 1:
        pool = Pool(args.n_jobs, initializer=_init_worker,
                    initargs=(train_set, val_set))
        print(f"[并行] 评估池已创建：{args.n_jobs} 进程")
    try:
        results: dict[str, dict] = {}

        # 1. 遗传算法
        if args.mode in ("ga", "all"):
            print(f"[GA] 开始优化：种群 {args.pop_size} × {args.generations} 代 ...")
            ga = run_ga(train_set, val_set, pop_size=args.pop_size,
                        generations=args.generations, seed=args.seed, pool=pool)
            params = decode(ga["best_genome"])

            best_params = {
                "C": params["C"], "gamma": params["gamma"], "kernel": params["kernel"],
                "conf_th": params["conf_th"], "prob_th": params["prob_th"],
                "fitness": round(ga["best_fitness"], 4),
            }
            (ARTIFACTS_DIR / "ga_best_params.json").write_text(
                json.dumps(best_params, ensure_ascii=False, indent=2), encoding="utf-8")
            (ARTIFACTS_DIR / "ga_history.json").write_text(
                json.dumps(ga["history"], ensure_ascii=False, indent=2), encoding="utf-8")
            plot_convergence(ga["history"], ARTIFACTS_DIR / "ga_convergence.png")
            save_final_model(best_params, train_set, val_set)

            results["ga"] = {"best_fitness": ga["best_fitness"], "params": best_params}
            print(f"[GA] 完成：最优 F1 = {ga['best_fitness']:.4f}")
            print(f"[GA] 最优参数: {best_params}")

        # 2. 随机搜索基线
        if args.mode in ("random", "all"):
            print(f"[随机搜索] 采样 {args.random_trials} 组参数 ...")
            rs = run_random_search(train_set, val_set, n_trials=args.random_trials,
                                   seed=args.seed, pool=pool)
            rs_out = {
                "best_fitness": rs["best_fitness"],
                "best_params": decode(rs["best_genome"]),
                "samples": rs["samples"],
            }
            (ARTIFACTS_DIR / "random_search_results.json").write_text(
                json.dumps(rs_out, ensure_ascii=False, indent=2), encoding="utf-8")
            results["random"] = {"best_fitness": rs["best_fitness"], "params": rs_out["best_params"]}
            print(f"[随机搜索] 最优 F1 = {rs['best_fitness']:.4f}")

        # 3. 网格搜索基线
        if args.mode in ("grid", "all"):
            print("[网格搜索] 4×3×2 组参数 ...")
            gs = run_grid_search(train_set, val_set, pool=pool)
            gs_out = {
                "best_fitness": gs["best_fitness"],
                "best_params": decode(gs["best_genome"]),
                "samples": gs["samples"],
            }
            (ARTIFACTS_DIR / "grid_search_results.json").write_text(
                json.dumps(gs_out, ensure_ascii=False, indent=2), encoding="utf-8")
            results["grid"] = {"best_fitness": gs["best_fitness"], "params": gs_out["best_params"]}
            print(f"[网格搜索] 最优 F1 = {gs['best_fitness']:.4f}")

        # 4. 对比输出
        if args.mode == "all":
            plot_comparison(results, ARTIFACTS_DIR / "ga_comparison.png")
            (ARTIFACTS_DIR / "ga_comparison.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")

        print("=" * 60)
        print("方法对比（验证集 F1）:")
        for key, val in results.items():
            print(f"  {key:>6}: {val['best_fitness']:.4f}")
        print("=" * 60)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    main()
