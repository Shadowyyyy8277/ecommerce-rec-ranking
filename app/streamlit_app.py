"""Streamlit demo for the recommendation system."""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_inference_features  # noqa: E402
from src.utils.io import load_config, read_json, resolve_path  # noqa: E402


st.set_page_config(
    page_title="E-commerce Recommendation Ranking Demo",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Render Streamlit recommendation demo."""
    _inject_styles()

    config = load_config()
    train_df = _load_train_df(config)
    registry = _load_registry(config)
    comparison = _load_model_comparison(config)
    metrics = _load_metrics(config)

    default_user = (
        str(train_df["user_id"].iloc[0])
        if train_df is not None and not train_df.empty
        else "u_000001"
    )

    _render_header()
    _render_metric_cards(metrics=metrics, registry=registry)

    with st.sidebar:
        st.markdown("### Demo Controls")
        st.caption("输入用户 ID 后，系统会从召回候选集中选择商品，并使用当前最佳排序模型进行 CTR 打分。")

        user_id = st.text_input("User ID", value=default_user)
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=30,
            value=int(config["serving"].get("default_top_k", 10)),
        )
        run_button = st.button("生成推荐", type="primary", use_container_width=True)

        st.divider()
        _render_model_status(registry=registry, metrics=metrics)

    tab_rec, tab_user, tab_metrics, tab_about = st.tabs(
        ["推荐结果", "用户画像", "模型与指标", "项目说明"]
    )

    with tab_rec:
        _render_recommendation_tab(
            user_id=user_id,
            top_k=top_k,
            config=config,
            run_button=run_button,
        )

    with tab_user:
        _render_user_profile_tab(user_id=user_id, train_df=train_df)

    with tab_metrics:
        _render_metrics_tab(comparison=comparison, metrics=metrics, registry=registry)

    with tab_about:
        _render_about_tab()


def _inject_styles() -> None:
    """Inject light-weight CSS for a cleaner portfolio-style UI."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }

        .hero-card {
            background: linear-gradient(135deg, #ffffff 0%, #f7f8fb 100%);
            border: 1px solid #eceff3;
            border-radius: 18px;
            padding: 24px 28px;
            margin-bottom: 18px;
            box-shadow: 0 2px 10px rgba(20, 32, 50, 0.04);
        }

        .hero-title {
            font-size: 32px;
            line-height: 1.25;
            font-weight: 800;
            margin-bottom: 8px;
            color: #1f2937;
        }

        .hero-subtitle {
            font-size: 15px;
            line-height: 1.7;
            color: #5f6b7a;
            margin-bottom: 14px;
        }

        .tag {
            display: inline-block;
            padding: 4px 10px;
            margin-right: 8px;
            margin-top: 4px;
            border-radius: 999px;
            border: 1px solid #e5e7eb;
            background: #ffffff;
            color: #374151;
            font-size: 12px;
            font-weight: 600;
        }

        .section-title {
            font-size: 20px;
            font-weight: 750;
            color: #1f2937;
            margin-top: 4px;
            margin-bottom: 10px;
        }

        .hint-text {
            color: #6b7280;
            font-size: 13px;
            line-height: 1.6;
        }

        .rec-card {
            background: #ffffff;
            border: 1px solid #eceff3;
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(20, 32, 50, 0.04);
        }

        .rec-topline {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .rec-rank {
            font-size: 18px;
            font-weight: 800;
            color: #111827;
        }

        .rec-item {
            font-size: 17px;
            font-weight: 750;
            color: #1f2937;
        }

        .rec-score {
            font-size: 13px;
            color: #4b5563;
            margin-top: 4px;
            margin-bottom: 6px;
        }

        .rec-reason {
            font-size: 13px;
            color: #6b7280;
            line-height: 1.6;
        }

        .small-note {
            font-size: 12px;
            color: #6b7280;
            line-height: 1.6;
        }

        .status-box {
            border: 1px solid #eceff3;
            border-radius: 14px;
            padding: 12px 14px;
            background: #ffffff;
            margin-bottom: 10px;
        }

        .status-label {
            font-size: 12px;
            color: #6b7280;
            margin-bottom: 2px;
        }

        .status-value {
            font-size: 15px;
            font-weight: 700;
            color: #1f2937;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    """Render top project header."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">电商推荐召回与 CTR 排序系统</div>
            <div class="hero-subtitle">
                一个面向电商推荐场景的端到端机器学习项目，覆盖数据处理、多路召回、排序建模、
                离线评估、实验记录和在线推理展示。当前 Demo 使用模拟用户行为数据，不涉及真实业务数据。
            </div>
            <span class="tag">Popular Recall</span>
            <span class="tag">ItemCF</span>
            <span class="tag">Item2Vec</span>
            <span class="tag">LR</span>
            <span class="tag">LightGBM</span>
            <span class="tag">DeepFM</span>
            <span class="tag">FastAPI</span>
            <span class="tag">Streamlit</span>
            <span class="tag">MLflow</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_cards(
    metrics: Dict[str, Any] | None,
    registry: Dict[str, Any] | None,
) -> None:
    """Render portfolio-style summary metrics."""
    best_model = "N/A"
    if metrics and metrics.get("best_model"):
        best_model = str(metrics["best_model"])
    elif registry and registry.get("best_model"):
        best_model = str(registry["best_model"])

    test_auc = _format_number(metrics.get("test_auc") if metrics else None, digits=4)
    test_logloss = _format_number(metrics.get("test_logloss") if metrics else None, digits=4)

    topk = metrics.get("topk", {}) if metrics else {}
    hit_rate_20 = topk.get("hit_rate@20", topk.get("hitrate@20"))
    hit_rate_20_text = _format_number(hit_rate_20, digits=4)

    latency = metrics.get("average_inference_latency_ms") if metrics else None
    latency_text = _format_number(latency, digits=4, suffix=" ms")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Best Model", best_model)

    with col2:
        st.metric("Test AUC", test_auc)

    with col3:
        st.metric("Test LogLoss", test_logloss)

    with col4:
        st.metric("HitRate@20", hit_rate_20_text)

    with col5:
        st.metric("Latency", latency_text)


def _render_model_status(
    registry: Dict[str, Any] | None,
    metrics: Dict[str, Any] | None,
) -> None:
    """Render sidebar status blocks."""
    model_name = "未加载"
    model_path = "N/A"
    model_size = "N/A"

    if registry:
        model_name = str(registry.get("best_model") or registry.get("model_type") or "N/A")
        model_path = str(registry.get("model_path", "N/A"))

    if metrics and metrics.get("model_size_mb") is not None:
        model_size = _format_number(metrics.get("model_size_mb"), digits=2, suffix=" MB")

    st.markdown(
        f"""
        <div class="status-box">
            <div class="status-label">Current Model</div>
            <div class="status-value">{escape(model_name)}</div>
        </div>
        <div class="status-box">
            <div class="status-label">Model Size</div>
            <div class="status-value">{escape(model_size)}</div>
        </div>
        <div class="small-note">
            Model registry: <br>
            <code>{escape(model_path)}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_recommendation_tab(
    user_id: str,
    top_k: int,
    config: Dict[str, Any],
    run_button: bool,
) -> None:
    """Render recommendation result area."""
    st.markdown('<div class="section-title">Top-K 推荐结果</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hint-text">
        点击“生成推荐”后，系统会读取该用户的召回候选集，构造推理特征，
        并使用当前最佳排序模型对候选商品进行 CTR 打分。
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not run_button:
        st.info("请在左侧输入 user_id 并点击“生成推荐”。")
        return

    with st.spinner("正在生成推荐结果..."):
        recs = _make_recommendations(user_id, top_k, config)

    if not recs:
        st.info("当前没有可展示的推荐。请先运行数据、召回和训练脚本，或检查 user_id 是否存在。")
        return

    st.success(f"已为用户 {user_id} 生成 Top-{len(recs)} 推荐结果。")

    left, right = st.columns([1.35, 1])

    with left:
        for rank, rec in enumerate(recs, start=1):
            _render_recommendation_card(rank, rec)

    with right:
        st.markdown('<div class="section-title">推荐结果表</div>', unsafe_allow_html=True)
        rec_df = pd.DataFrame(recs)
        if not rec_df.empty and "score" in rec_df.columns:
            rec_df["score"] = rec_df["score"].astype(float).round(6)
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">说明</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="hint-text">
            推荐分数表示排序模型对候选商品的点击/转化倾向预测值。
            当前 Demo 主要用于展示召回、排序和服务化流程，推荐理由为可解释性展示文案。
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_recommendation_card(rank: int, rec: Dict[str, Any]) -> None:
    """Render one recommendation item as a card."""
    item_id = escape(str(rec.get("item_id", "N/A")))
    score = _safe_float(rec.get("score"))
    reason = escape(str(rec.get("reason", "Ranked by CTR model.")))

    score_text = f"{score:.4f}" if score is not None else "N/A"

    st.markdown(
        f"""
        <div class="rec-card">
            <div class="rec-topline">
                <div class="rec-rank">#{rank}</div>
                <div class="rec-item">{item_id}</div>
            </div>
            <div class="rec-score">Predicted CTR Score: <b>{score_text}</b></div>
            <div class="rec-reason">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_user_profile_tab(user_id: str, train_df: pd.DataFrame | None) -> None:
    """Render user preference analysis."""
    st.markdown('<div class="section-title">用户历史偏好</div>', unsafe_allow_html=True)

    prefs = _user_preferences(user_id, train_df)
    if prefs is None:
        st.info("未找到该用户的历史行为数据。")
        return

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("历史价格中位数", f"{prefs['median_price']:.2f}")

    with metric_col2:
        st.metric("常看类目数", len(prefs["top_view_categories"]))

    with metric_col3:
        st.metric("购买类目数", len(prefs["top_buy_categories"]))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 常看类目 Top 5")
        st.dataframe(prefs["top_view_categories"], use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 购买类目 Top 5")
        st.dataframe(prefs["top_buy_categories"], use_container_width=True, hide_index=True)

    st.caption("用户画像基于训练集历史行为统计，仅用于 Demo 展示。")


def _render_metrics_tab(
    comparison: pd.DataFrame | None,
    metrics: Dict[str, Any] | None,
    registry: Dict[str, Any] | None,
) -> None:
    """Render model comparison and evaluation metrics."""
    st.markdown('<div class="section-title">模型对比</div>', unsafe_allow_html=True)

    if comparison is not None:
        display_df = comparison.copy()
        for col in display_df.columns:
            if pd.api.types.is_float_dtype(display_df[col]):
                display_df[col] = display_df[col].round(4)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("尚未生成模型对比表。请先运行 python scripts/run_train_all.py 和 python scripts/run_evaluate.py。")

    st.markdown('<div class="section-title">最终评估结果</div>', unsafe_allow_html=True)

    if metrics is not None:
        topk = metrics.get("topk", {})
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Precision@20", _format_number(topk.get("precision@20"), digits=4))

        with c2:
            st.metric("Recall@20", _format_number(topk.get("recall@20"), digits=4))

        with c3:
            st.metric("NDCG@20", _format_number(topk.get("ndcg@20"), digits=4))

        with c4:
            st.metric(
                "HitRate@20",
                _format_number(topk.get("hit_rate@20", topk.get("hitrate@20")), digits=4),
            )

        with st.expander("查看完整 metrics.json"):
            st.json(metrics)
    else:
        st.info("尚未生成最终评估结果。")

    if registry is not None:
        with st.expander("查看 model_registry.json"):
            st.json(registry)


def _render_about_tab() -> None:
    """Render project explanation."""
    st.markdown('<div class="section-title">项目定位</div>', unsafe_allow_html=True)
    st.markdown(
        """
        本项目是一个面向电商推荐场景的个人实践项目，重点复现工业推荐系统中的
        “召回 - 排序 - 评估 - 服务化”闭环流程。数据为模拟用户行为数据，
        不涉及真实业务数据。
        """
    )

    st.markdown('<div class="section-title">核心流程</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. **数据与特征工程**：构造用户、商品、上下文和历史统计特征。  
        2. **多路召回**：使用 Popular Recall、ItemCF、Item2Vec 生成候选商品。  
        3. **CTR 排序**：对比 LR、LightGBM、DeepFM 三类排序模型。  
        4. **离线评估**：使用 AUC、LogLoss、Precision@K、Recall@K、NDCG@K、HitRate@K。  
        5. **在线推理**：通过 FastAPI 和 Streamlit 展示模型服务化效果。  
        """
    )

    st.markdown('<div class="section-title">项目边界</div>', unsafe_allow_html=True)
    st.markdown(
        """
        当前 Demo 为单机离线实验与本地服务展示，未接入真实线上特征服务、消息队列、
        Redis 缓存、A/B 实验平台和大规模分布式训练。后续可继续扩展冷启动策略、
        多目标排序、在线特征更新和更复杂的深度推荐模型。
        """
    )


def _make_recommendations(user_id: str, top_k: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create recommendations directly in Streamlit, with popular fallback."""
    registry = _load_registry(config)
    feature_config = _load_feature_config(config)
    train_df = _load_train_df(config)

    if registry and feature_config and train_df is not None:
        model_path = resolve_path(registry.get("model_path", ""))
        if model_path.exists():
            try:
                recall_path = resolve_path(config["paths"]["recall_dir"]) / "merged_recall.csv"
                if recall_path.exists():
                    recall_df = pd.read_csv(recall_path)
                    candidates = recall_df[recall_df["user_id"] == user_id].copy()
                else:
                    candidates = pd.DataFrame()

                if candidates.empty:
                    candidates = pd.DataFrame(
                        {
                            "user_id": user_id,
                            "item_id": _popular_items(train_df, top_k * 5),
                            "recall_score": 1.0,
                            "recall_type": "popular",
                        }
                    )

                features = build_inference_features(
                    candidates,
                    train_df,
                    config=config,
                    feature_config=feature_config,
                )
                features["score"] = _predict_by_model(
                    registry["model_type"],
                    model_path,
                    features,
                    feature_config,
                )

                ranked = (
                    features.groupby(["user_id", "item_id"], as_index=False)["score"]
                    .max()
                    .sort_values("score", ascending=False)
                    .head(top_k)
                )

                model_name = str(registry.get("best_model") or registry.get("model_type") or "model")
                results: List[Dict[str, Any]] = []
                for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                    score = float(row.score)
                    results.append(
                        {
                            "item_id": row.item_id,
                            "score": score,
                            "reason": _build_recommend_reason(
                                rank=rank,
                                score=score,
                                model_name=model_name,
                            ),
                        }
                    )
                return results

            except Exception as exc:
                st.warning(f"模型排序失败，已切换 popular fallback：{exc}")

    if train_df is None:
        return []

    return [
        {
            "item_id": item_id,
            "score": 0.0,
            "reason": "模型或召回结果不可用，当前使用全局热门商品作为兜底推荐。",
        }
        for item_id in _popular_items(train_df, top_k)
    ]


def _build_recommend_reason(rank: int, score: float, model_name: str) -> str:
    """Generate a concise recommendation reason for display."""
    model_label = model_name.upper() if model_name else "MODEL"

    if rank == 1:
        return f"{model_label} 在召回候选集中给出最高 CTR 预估分，因此排在首位。"

    if score >= 0.25:
        return "该商品在多路召回候选集中排序分较高，模型判断具备较强点击倾向。"

    if score >= 0.20:
        return "该商品通过召回融合进入候选集，并在 CTR 排序阶段保持较高得分。"

    return "该商品在候选集中通过排序模型筛选进入 Top-K 推荐列表。"


def _user_preferences(user_id: str, train_df: pd.DataFrame | None) -> Dict[str, Any] | None:
    """Summarize a user's historical category and price preferences."""
    if train_df is None:
        return None

    user_hist = train_df[train_df["user_id"] == user_id]
    if user_hist.empty:
        return None

    top_view = user_hist["category_id"].value_counts().head(5).reset_index()
    top_view.columns = ["category_id", "count"]

    buy_hist = user_hist[user_hist["behavior_type"] == "buy"]
    top_buy = buy_hist["category_id"].value_counts().head(5).reset_index()
    top_buy.columns = ["category_id", "buy_count"]

    return {
        "top_view_categories": top_view,
        "top_buy_categories": top_buy,
        "median_price": float(user_hist["item_price"].median()),
    }


def _predict_by_model(
    model_type: str,
    model_path: Path,
    df: pd.DataFrame,
    feature_config: Dict[str, Any],
) -> pd.Series:
    """Predict CTR with lazy imports."""
    if model_type == "lr":
        from src.models.train_lr import predict_lr

        return predict_lr(model_path, df, feature_config)

    if model_type == "lightgbm":
        from src.models.train_lightgbm import predict_lightgbm

        return predict_lightgbm(model_path, df, feature_config)

    if model_type == "deepfm":
        from src.models.train_deepfm import predict_deepfm

        return predict_deepfm(model_path, df, feature_config)

    raise ValueError(f"Unsupported model_type: {model_type}")


def _load_registry(config: Dict[str, Any]) -> Dict[str, Any] | None:
    """Load local model selection metadata."""
    path = resolve_path(config["paths"]["model_registry"])
    if not path.exists() or path.read_text(encoding="utf-8").strip() in {"", "{}"}:
        return None
    return read_json(path)


def _load_feature_config(config: Dict[str, Any]) -> Dict[str, Any] | None:
    """Load saved feature column metadata."""
    path = resolve_path(config["paths"]["processed_dir"]) / "feature_columns.json"
    return read_json(path) if path.exists() else None


def _load_train_df(config: Dict[str, Any]) -> pd.DataFrame | None:
    """Load train split if available."""
    path = resolve_path(config["paths"]["processed_dir"]) / "train.csv"
    return pd.read_csv(path) if path.exists() else None


def _load_model_comparison(config: Dict[str, Any]) -> pd.DataFrame | None:
    """Load model comparison CSV."""
    path = resolve_path(config["paths"]["metrics_dir"]) / "model_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


def _load_metrics(config: Dict[str, Any]) -> Dict[str, Any] | None:
    """Load final evaluation metrics JSON."""
    path = resolve_path(config["paths"]["metrics_dir"]) / "metrics.json"
    return read_json(path) if path.exists() else None


def _popular_items(train_df: pd.DataFrame, top_k: int) -> List[str]:
    """Return globally popular items from train split."""
    return train_df["item_id"].value_counts().head(top_k).index.astype(str).tolist()


def _format_number(value: Any, digits: int = 4, suffix: str = "") -> str:
    """Format numeric metric values safely."""
    number = _safe_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}{suffix}"


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()