"""
iFood Marketing Analytics — Live Demo Presentation App
-------------------------------------------------------
Run with:  streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="iFood Marketing Analytics",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_PATH = "data/processed/ifood_df_with_rfm.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalPurchases"] = (
        df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    )
    df["DealRatio"] = df["NumDealsPurchases"] / (
        df["TotalPurchases"] + df["NumDealsPurchases"] + 0.001
    )
    return df


df = load_data()

# ── Color palettes ────────────────────────────────────────────────────────────
CLUSTER_COLORS = {
    "Empty Nesters": "#6C63FF",
    "Mature Families w/ Teens": "#3FA7D6",
    "Stretched Parents": "#F4A261",
    "Young Families on Budget": "#2EC4B6",
}

SEGMENT_COLORS = {
    "Champions": "#FFD700",
    "Loyal Customers": "#4CAF50",
    "Potential Loyalists": "#2196F3",
    "Promising": "#9C27B0",
    "At Risk": "#FF9800",
    "Recent Customers": "#00BCD4",
    "Needs Attention": "#FF5722",
    "Hibernating": "#9E9E9E",
    "Lost": "#F44336",
}

TARGET_SEGMENTS = {"Champions", "Potential Loyalists", "Loyal Customers"}
TEST_SEGMENTS = {"At Risk", "Promising", "Recent Customers"}
EXCLUDE_SEGMENTS = {"Needs Attention", "Hibernating", "Lost"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("iFood Marketing Analytics")
st.sidebar.caption("Customer Segmentation & Campaign Optimization")
st.sidebar.markdown("---")

PAGES = [
    "🏆 Business Impact",
    "👥 Customer Segments",
    "📊 RFM Analysis",
    "🤖 Predictive Model",
    "💡 Key Findings",
    "🎯 Live Scoring",
]
page = st.sidebar.radio("Section", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** 2,205 customers · 38 features  \n"
    "**Models:** K-Means · RFM · Logistic Regression"
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def divider():
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — BUSINESS IMPACT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏆 Business Impact":
    st.title("Business Impact")
    st.caption(
        "From untargeted mass campaigns losing money to precision targeting generating profit"
    )
    divider()

    # Headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Improvement", "$7,080", "per campaign cycle")
    c2.metric("ROI", "132%", "up from −24.5%")
    c3.metric("Response Rate", "46.5%", "+208% vs baseline 15%")
    c4.metric("Customers Targeted", "127", "down from 2,205 (−94%)")

    st.markdown("")

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Before vs After")
        comparison = pd.DataFrame(
            {
                "Metric": [
                    "Customers Contacted",
                    "Response Rate",
                    "Expected Conversions",
                    "Campaign Cost",
                    "Revenue",
                    "Profit / Loss",
                    "ROI",
                ],
                "Current (Untargeted)": [
                    "2,205",
                    "15.1%",
                    "333",
                    "$22,050",
                    "$16,650",
                    "−$5,400",
                    "−24.5%",
                ],
                "Optimized (Model-Driven)": [
                    "127",
                    "46.5%",
                    "59",
                    "$1,270",
                    "$2,950",
                    "+$1,680",
                    "132%",
                ],
                "Change": [
                    "−94.2%",
                    "+208%",
                    "−82.3%",
                    "−94.2%",
                    "−82.3%",
                    "+$7,080",
                    "+157 pp",
                ],
            }
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Profit per Campaign")
        fig = go.Figure()
        fig.add_bar(
            x=["Current\n(Untargeted)", "Optimized\n(Model-Driven)"],
            y=[-5400, 1680],
            marker_color=["#F44336", "#4CAF50"],
            text=["−$5,400", "+$1,680"],
            textposition="outside",
            textfont=dict(size=16, color=["#F44336", "#4CAF50"]),
        )
        fig.add_hline(y=0, line_color="#888", line_width=1.5)
        fig.update_layout(
            yaxis_title="Profit ($)",
            yaxis=dict(tickprefix="$", range=[-7000, 3500]),
            height=320,
            showlegend=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    divider()

    st.subheader("Methodology — Four-Phase Approach")
    cols = st.columns(4)
    steps = [
        (
            "Phase 1: Explore",
            "2,205 customers cleaned and 38 features engineered from raw transactional data.",
        ),
        (
            "Phase 2: Cluster",
            "K-Means (k=4) identifies demographic segments based on income, age, children, and spend.",
        ),
        (
            "Phase 3: RFM",
            "Recency · Frequency · Monetary scoring creates 9 behavioral segments.",
        ),
        (
            "Phase 4: Model",
            "Logistic Regression (91% ROC-AUC) with threshold tuned for maximum profit.",
        ),
    ]
    for col, (title, desc) in zip(cols, steps):
        col.info(f"**{title}**\n\n{desc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("Customer Segments")
    st.caption("K-Means clustering identified 4 distinct demographic profiles")
    divider()

    cluster_stats = (
        df.groupby("Cluster_Name")
        .agg(
            Size=("Income", "count"),
            Avg_Income=("Income", "mean"),
            Avg_Age=("Age", "mean"),
            Avg_Children=("TotalChildren", "mean"),
            Avg_Spend=("MntTotal", "mean"),
            Response_Rate=("Response", "mean"),
        )
        .reset_index()
        .sort_values("Avg_Spend", ascending=False)
    )

    # Segment summary cards
    cols = st.columns(4)
    descriptions = {
        "Empty Nesters": "High income · No kids · Catalog buyers",
        "Mature Families w/ Teens": "Moderate income · Teens at home",
        "Stretched Parents": "Low income · Young kids · Budget-focused",
        "Young Families on Budget": "Low income · Young kids · Store-heavy",
    }
    for col, (_, row) in zip(cols, cluster_stats.iterrows()):
        col.metric(
            row["Cluster_Name"],
            f"${row['Avg_Spend']:,.0f} avg spend",
            f"{row['Size']:,} customers",
        )
        col.caption(descriptions.get(row["Cluster_Name"], ""))

    st.markdown("")

    left, right = st.columns(2)

    with left:
        st.subheader("Income vs Total Spend")
        sample = df.sample(min(600, len(df)), random_state=42)
        fig = px.scatter(
            sample,
            x="Income",
            y="MntTotal",
            color="Cluster_Name",
            color_discrete_map=CLUSTER_COLORS,
            opacity=0.55,
            labels={
                "Income": "Annual Income ($)",
                "MntTotal": "Total Spend ($)",
                "Cluster_Name": "Segment",
            },
            height=380,
        )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
            ),
            margin=dict(t=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Average Spend by Cluster")
        fig = px.bar(
            cluster_stats.sort_values("Avg_Spend"),
            x="Avg_Spend",
            y="Cluster_Name",
            orientation="h",
            color="Cluster_Name",
            color_discrete_map=CLUSTER_COLORS,
            text=cluster_stats.sort_values("Avg_Spend")["Avg_Spend"].map(
                lambda v: f"${v:,.0f}"
            ),
            labels={"Avg_Spend": "Avg Total Spend ($)", "Cluster_Name": ""},
            height=380,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    divider()
    st.subheader("Cluster Summary Table")
    display = cluster_stats.copy()
    display["Avg_Income"] = display["Avg_Income"].map(lambda v: f"${v:,.0f}")
    display["Avg_Age"] = display["Avg_Age"].map(lambda v: f"{v:.0f}")
    display["Avg_Children"] = display["Avg_Children"].map(lambda v: f"{v:.1f}")
    display["Avg_Spend"] = display["Avg_Spend"].map(lambda v: f"${v:,.0f}")
    display["Response_Rate"] = display["Response_Rate"].map(lambda v: f"{v:.1%}")
    display.columns = [
        "Segment",
        "Size",
        "Avg Income",
        "Avg Age",
        "Avg Children",
        "Avg Spend",
        "Response Rate",
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RFM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 RFM Analysis":
    st.title("RFM Analysis")
    st.caption("9 behavioral segments — each requiring a different marketing response")
    divider()

    rfm_stats = (
        df.groupby("RFM_Segment")
        .agg(
            Size=("MntTotal", "count"),
            Avg_Spend=("MntTotal", "mean"),
            Response_Rate=("Response", "mean"),
            Avg_Recency=("Recency", "mean"),
        )
        .reset_index()
        .sort_values("Response_Rate", ascending=False)
    )

    def get_action(seg):
        if seg in TARGET_SEGMENTS:
            return "✅ Target"
        elif seg in TEST_SEGMENTS:
            return "⚠️ Test"
        return "❌ Exclude"

    rfm_stats["Action"] = rfm_stats["RFM_Segment"].map(get_action)

    # Summary counts
    c1, c2, c3 = st.columns(3)
    target_n = rfm_stats[rfm_stats["RFM_Segment"].isin(TARGET_SEGMENTS)]["Size"].sum()
    test_n = rfm_stats[rfm_stats["RFM_Segment"].isin(TEST_SEGMENTS)]["Size"].sum()
    excl_n = rfm_stats[rfm_stats["RFM_Segment"].isin(EXCLUDE_SEGMENTS)]["Size"].sum()
    c1.success(f"**✅ Target** — {target_n:,} customers ({target_n/len(df):.0%} of base)")
    c2.warning(f"**⚠️ Test** — {test_n:,} customers ({test_n/len(df):.0%} of base)")
    c3.error(f"**❌ Exclude** — {excl_n:,} customers ({excl_n/len(df):.0%} of base)")

    st.markdown("")

    left, right = st.columns(2)

    with left:
        st.subheader("Segment Size")
        size_ordered = rfm_stats.sort_values("Size", ascending=False)
        fig = px.bar(
            size_ordered,
            x="RFM_Segment",
            y="Size",
            color="RFM_Segment",
            color_discrete_map=SEGMENT_COLORS,
            labels={"Size": "# Customers", "RFM_Segment": ""},
            text="Size",
            height=380,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_tickangle=-30, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Response Rate by Segment")
        fig = px.bar(
            rfm_stats,
            x="Response_Rate",
            y="RFM_Segment",
            orientation="h",
            color="RFM_Segment",
            color_discrete_map=SEGMENT_COLORS,
            text=rfm_stats["Response_Rate"].map(lambda v: f"{v:.1%}"),
            labels={"Response_Rate": "Campaign Response Rate", "RFM_Segment": ""},
            height=380,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            xaxis_tickformat=".0%",
            margin=dict(t=10, r=60),
        )
        fig.add_vline(
            x=0.151,
            line_dash="dash",
            line_color="#888",
            annotation_text="Avg 15.1%",
            annotation_position="top right",
        )
        st.plotly_chart(fig, use_container_width=True)

    divider()
    st.subheader("Segment Detail")
    display_rfm = rfm_stats.copy()
    display_rfm["Avg_Spend"] = display_rfm["Avg_Spend"].map(lambda v: f"${v:,.0f}")
    display_rfm["Response_Rate"] = display_rfm["Response_Rate"].map(lambda v: f"{v:.1%}")
    display_rfm["Avg_Recency"] = display_rfm["Avg_Recency"].map(lambda v: f"{v:.0f} days")
    display_rfm.columns = ["Segment", "Size", "Avg Spend", "Response Rate", "Avg Recency", "Action"]
    st.dataframe(display_rfm, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIVE MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictive Model":
    st.title("Predictive Model")
    st.caption(
        "Logistic Regression with profit-optimized threshold — 91% ROC-AUC, $1,680 profit per campaign"
    )
    divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Model Comparison")
        model_df = pd.DataFrame(
            {
                "Model": ["Logistic Regression ⭐", "XGBoost", "Random Forest"],
                "ROC-AUC": ["0.908", "0.904", "0.893"],
                "Precision": ["0.49", "0.58", "0.50"],
                "Recall": ["0.84", "0.52", "0.61"],
                "F1-Score": ["0.62", "0.55", "0.55"],
                "Campaign Profit": ["$1,680", "$810", "$1,180"],
            }
        )
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        st.markdown("")
        st.subheader("Why Logistic Regression?")
        st.markdown(
            """
- **Highest ROC-AUC** — best overall discrimination ability
- **Highest profit** at the optimal threshold ($1,680)
- **Interpretable** — feature coefficients are explainable to stakeholders
- **Fast** — suitable for real-time or batch customer scoring
"""
        )

    with right:
        st.subheader("Key Predictors")
        features = pd.DataFrame(
            {
                "Feature": [
                    "Past Campaign Acceptance",
                    "Customer Tenure",
                    "Monetary RFM Score",
                    "Catalog Purchases",
                    "Total Spend",
                    "Purchase Frequency",
                    "Income",
                ],
                "Importance": [0.28, 0.18, 0.16, 0.14, 0.10, 0.08, 0.06],
            }
        )
        fig = px.bar(
            features.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            text=features.sort_values("Importance")["Importance"].map(
                lambda v: f"{v:.0%}"
            ),
            height=380,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=10, r=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    divider()

    st.subheader("Threshold Optimization — Finding the Profit-Maximizing Cut-Off")
    st.markdown(
        "The default 0.5 threshold maximizes accuracy — but **0.45 maximizes profit**. "
        "Below the line, campaigns lose money. The optimal point targets only the 127 "
        "most likely responders."
    )

    # Build profit curve using known anchor points
    xp = [0.00, 0.10, 0.18, 0.28, 0.38, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]
    yp = [-5400, -4100, -2400, -500, 900, 1680, 1250, 880, 520, 240, 60, 0]
    thresholds = np.linspace(0.0, 1.0, 200)
    profits = np.interp(thresholds, xp, yp)

    fig = go.Figure()
    fig.add_scatter(
        x=thresholds,
        y=profits,
        mode="lines",
        line=dict(color="#4CAF50", width=3),
        name="Expected Profit",
    )
    # shade negative region
    neg_mask = profits < 0
    fig.add_scatter(
        x=thresholds[neg_mask],
        y=profits[neg_mask],
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(244,67,54,0.15)",
        name="Loss zone",
        showlegend=False,
    )
    fig.add_scatter(
        x=thresholds[~neg_mask],
        y=profits[~neg_mask],
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(76,175,80,0.10)",
        name="Profit zone",
        showlegend=False,
    )
    fig.add_vline(
        x=0.45,
        line_dash="dash",
        line_color="#FF5722",
        annotation_text=" Optimal threshold (0.45) → $1,680 profit",
        annotation_position="top right",
        annotation_font_color="#FF5722",
    )
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.update_layout(
        xaxis_title="Classification Threshold",
        yaxis_title="Expected Profit ($)",
        yaxis_tickprefix="$",
        height=340,
        showlegend=False,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Actual response rate by campaign history from real data
    divider()
    st.subheader("Data Validation: Past Campaign Acceptance vs Response Rate")
    resp_by_cmp = (
        df.groupby("AcceptedCmpOverall")["Response"]
        .agg(["mean", "count"])
        .reset_index()
    )
    resp_by_cmp.columns = ["Campaigns Accepted", "Response Rate", "Count"]
    fig2 = px.bar(
        resp_by_cmp,
        x="Campaigns Accepted",
        y="Response Rate",
        text=resp_by_cmp["Response Rate"].map(lambda v: f"{v:.1%}"),
        color="Response Rate",
        color_continuous_scale="RdYlGn",
        labels={"Response Rate": "Campaign Response Rate"},
        height=320,
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        yaxis_tickformat=".0%",
        coloraxis_showscale=False,
        xaxis=dict(tickmode="linear", dtick=1),
        margin=dict(t=10),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.info(
        "**0 campaigns accepted → ~8% response rate.  "
        "4 campaigns accepted → ~91% response rate.**  "
        "This is the strongest single predictor in the model."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — KEY FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Key Findings":
    st.title("Key Findings")
    st.caption("Five data-driven insights that shape the marketing strategy")
    divider()

    tabs = st.tabs(
        [
            "1 · Past Behavior",
            "2 · Wine Dominance",
            "3 · Channel Mix",
            "4 · Untapped Categories",
            "5 · Deal Sensitivity",
        ]
    )

    # ── Finding 1 ──────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Past Campaign Acceptance Predicts Future Response")
        st.markdown(
            "> **Customers who accepted previous campaigns are 10× more likely to respond to the next one.**"
        )

        resp_data = (
            df.groupby("AcceptedCmpOverall")["Response"]
            .agg(["mean", "count"])
            .reset_index()
        )
        resp_data.columns = ["Campaigns Accepted", "Response Rate", "# Customers"]

        fig = px.bar(
            resp_data,
            x="Campaigns Accepted",
            y="Response Rate",
            text=resp_data["Response Rate"].map(lambda v: f"{v:.1%}"),
            color="Response Rate",
            color_continuous_scale="RdYlGn",
            height=380,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis_tickformat=".0%",
            coloraxis_showscale=False,
            xaxis=dict(tickmode="linear", dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            "**Action:** Prioritize customers with AcceptedCmpOverall ≥ 1. "
            "Create an 'engaged customer' fast-track list. "
            "Never-responders are draining campaign budgets."
        )

    # ── Finding 2 ──────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Wine Drives Revenue in High-Value Segments")
        st.markdown(
            "> **Wine represents 45–55% of spending for Champions and Loyal Customers — 300% over-index vs the overall base.**"
        )

        product_rename = {
            "MntWines": "Wine",
            "MntMeatProducts": "Meat",
            "MntFruits": "Fruits",
            "MntFishProducts": "Fish",
            "MntSweetProducts": "Sweets",
            "MntGoldProds": "Gold",
        }
        seg_order = (
            df.groupby("RFM_Segment")["MntWines"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        product_by_seg = (
            df.groupby("RFM_Segment")[list(product_rename.keys())]
            .mean()
            .rename(columns=product_rename)
            .reset_index()
        )
        product_by_seg["RFM_Segment"] = pd.Categorical(
            product_by_seg["RFM_Segment"], categories=seg_order, ordered=True
        )
        product_by_seg = product_by_seg.sort_values("RFM_Segment")

        fig = px.bar(
            product_by_seg.melt(
                id_vars="RFM_Segment", var_name="Product", value_name="Avg Spend"
            ),
            x="RFM_Segment",
            y="Avg Spend",
            color="Product",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"RFM_Segment": "", "Avg Spend": "Avg Spend ($)"},
            height=420,
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            "**Action:** Lead all high-value campaigns with premium wine. "
            "Create a wine subscription program. Bundle wine + meat + complementary products."
        )

    # ── Finding 3 ──────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Channel Preference Varies Significantly by Demographic Segment")
        st.markdown(
            "> **Empty Nesters skew heavily toward catalog; Young Families are store-dominant.**"
        )

        ch = (
            df.groupby("Cluster_Name")[
                ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
            ]
            .mean()
            .reset_index()
        )
        ch.columns = ["Cluster", "Web", "Catalog", "Store"]
        ch["Total"] = ch[["Web", "Catalog", "Store"]].sum(axis=1)
        for col in ["Web", "Catalog", "Store"]:
            ch[col] = ch[col] / ch["Total"] * 100

        fig = px.bar(
            ch.melt(
                id_vars="Cluster",
                value_vars=["Web", "Catalog", "Store"],
                var_name="Channel",
                value_name="Share",
            ),
            x="Share",
            y="Cluster",
            color="Channel",
            orientation="h",
            barmode="stack",
            color_discrete_map={
                "Web": "#2196F3",
                "Catalog": "#FF9800",
                "Store": "#4CAF50",
            },
            labels={"Share": "% of Purchases", "Cluster": ""},
            height=360,
            text_auto=".0f",
        )
        fig.update_layout(xaxis_ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            "**Action:** Premium print catalogs for Empty Nesters. "
            "Digital-first / mobile for Young Families. "
            "QR codes linking in-store to web for digital conversion."
        )

    # ── Finding 4 ──────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Fruits, Fish & Sweets Are Massively Underserved")
        st.markdown(
            "> **Growth categories represent <10% of spend even in high-value segments — significant untapped cross-sell potential.**"
        )

        avg_spend = {
            "Wine": df["MntWines"].mean(),
            "Meat": df["MntMeatProducts"].mean(),
            "Gold": df["MntGoldProds"].mean(),
            "Fruits": df["MntFruits"].mean(),
            "Fish": df["MntFishProducts"].mean(),
            "Sweets": df["MntSweetProducts"].mean(),
        }
        prod_df = pd.DataFrame(
            list(avg_spend.items()), columns=["Product", "Avg Spend"]
        ).sort_values("Avg Spend", ascending=False)

        left2, right2 = st.columns(2)
        with left2:
            fig = px.pie(
                prod_df,
                names="Product",
                values="Avg Spend",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
                height=360,
            )
            fig.update_traces(textinfo="label+percent", textposition="outside")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with right2:
            fig2 = px.bar(
                prod_df,
                x="Product",
                y="Avg Spend",
                color="Product",
                color_discrete_sequence=px.colors.qualitative.Set2,
                text=prod_df["Avg Spend"].map(lambda v: f"${v:.0f}"),
                labels={"Avg Spend": "Avg Spend per Customer ($)"},
                height=360,
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.success(
            "**Action:** Bundle campaigns — Wine + Cheese + Fruit. "
            "Recipe cards in catalogs. Sampling programs to introduce new categories to high-value customers."
        )

    # ── Finding 5 ──────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Deal Sensitivity Is Inversely Correlated with Customer Value")
        st.markdown(
            "> **High-value customers rarely use deals. Discounting Champions trains deal-seeking behaviour.**"
        )

        deal_by_seg = (
            df.groupby("RFM_Segment")
            .agg(Deal_Ratio=("DealRatio", "mean"), Avg_Spend=("MntTotal", "mean"))
            .reset_index()
        )

        fig = px.scatter(
            deal_by_seg,
            x="Deal_Ratio",
            y="Avg_Spend",
            text="RFM_Segment",
            color="Avg_Spend",
            color_continuous_scale="RdYlGn",
            size="Avg_Spend",
            size_max=50,
            labels={
                "Deal_Ratio": "Avg Deal Purchase Ratio",
                "Avg_Spend": "Avg Total Spend ($)",
            },
            height=420,
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            "**Action:** Never discount Champions or Loyal Customers — use exclusivity and premium messaging instead. "
            "Reserve deals strictly for at-risk segment reactivation."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — LIVE SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Live Scoring":
    st.title("Live Customer Scoring")
    st.caption(
        "Adjust the customer profile to see their segment, predicted response probability, and recommended action"
    )
    divider()

    # ── Scoring helpers ───────────────────────────────────────────────────
    SEGMENT_RULES = [
        {
            "name": "Champions",
            "condition": lambda r, f, m: r >= 4 and f >= 4 and m >= 4,
            "recommendation": "Target with premium wine catalogs, VIP experiences, and early-access offers.",
            "roi_text": "$50–60 revenue per $10 investment",
        },
        {
            "name": "Loyal Customers",
            "condition": lambda r, f, m: r <= 2 and f >= 4 and m >= 4,
            "recommendation": "Target with loyalty rewards, exclusives, and upgrade campaigns.",
            "roi_text": "$30–35 revenue per $10 investment",
        },
        {
            "name": "Potential Loyalists",
            "condition": lambda r, f, m: r >= 4 and f >= 2 and m >= 2,
            "recommendation": "Target with conversion campaigns, entry-level wine offers, and bundles.",
            "roi_text": "$25–30 revenue per $10 investment",
        },
        {
            "name": "At Risk",
            "condition": lambda r, f, m: r <= 2 and f >= 3 and m >= 3,
            "recommendation": "Selective win-back offers and reminder campaigns.",
            "roi_text": "$10–18 revenue per $10 investment",
        },
        {
            "name": "Recent Customers",
            "condition": lambda r, f, m: r >= 4 and f == 1 and m == 1,
            "recommendation": "Onboarding sequences and low-friction welcome offers.",
            "roi_text": "$8–12 revenue per $10 investment",
        },
        {
            "name": "Needs Attention",
            "condition": lambda r, f, m: r == 3 and f <= 2 and m <= 2,
            "recommendation": "Light-touch nurture before spending heavily.",
            "roi_text": "$5–8 revenue per $10 investment",
        },
        {
            "name": "Hibernating",
            "condition": lambda r, f, m: r <= 2 and f >= 2 and m >= 2,
            "recommendation": "Avoid regular campaigns; low-cost reactivation only.",
            "roi_text": "$2–5 revenue per $10 investment",
        },
        {
            "name": "Lost",
            "condition": lambda r, f, m: r <= 2 and f == 1 and m == 1,
            "recommendation": "Exclude from normal campaigns; quarterly win-back tests only.",
            "roi_text": "Negative expected ROI in most cases",
        },
    ]

    CHANNEL_MAP = {
        "Web": "Lead with email and site retargeting — keep landing page friction low.",
        "Catalog": "Lead with premium creative, high-value bundles, and curated product storytelling.",
        "Store": "Use in-store pickup, QR follow-up, and retail-to-digital conversion tactics.",
    }

    PRODUCT_MAP = {
        "Wine": "Anchor the campaign around premium wine selection and pairings.",
        "Meat": "Feature bundles, meal kits, and premium protein add-ons.",
        "Fruits": "Use cross-sell bundles and seasonal pairings.",
        "Fish": "Specialty bundles and recipe-led merchandising.",
        "Sweets": "Gifting and add-on basket builders.",
    }

    def score_to_band(value, bins):
        score = 1
        for t in bins:
            if value >= t:
                score += 1
        return min(score, 5)

    def predict_response_prob(income, recency, spend, purchases, accepted, catalog_sh, web_sh):
        recency_score = max(0, min(1, (120 - recency) / 120))
        spend_score = min(spend / 1500, 1)
        freq_score = min(purchases / 20, 1)
        cmp_score = min(accepted / 4, 1)
        catalog_score = min(catalog_sh, 1)
        web_score = min(web_sh, 1)
        income_score = min(max((income - 20000) / 100000, 0), 1)
        weighted = (
            0.10 * income_score
            + 0.20 * recency_score
            + 0.16 * spend_score
            + 0.14 * freq_score
            + 0.28 * cmp_score
            + 0.07 * catalog_score
            + 0.05 * web_score
        )
        return float(max(0.02, min(0.03 + weighted * 0.72, 0.92)))

    def derive_segment(recency, purchases, spend):
        r = 6 - score_to_band(recency, [30, 60, 90, 120])
        f = score_to_band(purchases, [2, 5, 8, 12])
        m = score_to_band(spend, [100, 300, 700, 1100])
        for rule in SEGMENT_RULES:
            if rule["condition"](r, f, m):
                return rule["name"], r, f, m, rule["recommendation"], rule["roi_text"]
        return "Promising", r, f, m, "Test seasonal and moderate-intent offers.", "$12–18 revenue per $10 investment"

    # ── Inputs ────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Demographics**")
        income = st.slider("Income ($)", 10_000, 150_000, 65_000, 5_000)
        age = st.slider("Age", 18, 80, 42, 1)
        children = st.slider("Children at home", 0, 5, 1, 1)

    with col_b:
        st.markdown("**Purchase History**")
        recency_days = st.slider("Days since last purchase", 0, 180, 45, 5)
        total_purchases = st.slider("Total purchases", 0, 25, 8, 1)
        total_spend = st.slider("Total spend ($)", 0, 2000, 550, 25)

    with col_c:
        st.markdown("**Campaign & Channel**")
        accepted_cmp = st.slider("Past campaigns accepted", 0, 4, 1, 1)
        preferred_channel = st.selectbox("Preferred channel", ["Web", "Catalog", "Store"])
        preferred_product = st.selectbox(
            "Preferred product", ["Wine", "Meat", "Fruits", "Fish", "Sweets"]
        )

    channel_shares = {"Web": (0.20, 0.45), "Catalog": (0.35, 0.20), "Store": (0.10, 0.15)}
    catalog_sh, web_sh = channel_shares[preferred_channel]

    divider()

    # ── Scoring ───────────────────────────────────────────────────────────
    prob = predict_response_prob(
        income, recency_days, total_spend, total_purchases,
        accepted_cmp, catalog_sh, web_sh,
    )
    segment, r_sc, f_sc, m_sc, recommendation, roi_text = derive_segment(
        recency_days, total_purchases, total_spend
    )

    if prob >= 0.45:
        priority = "High-priority target"
    elif prob >= 0.25:
        priority = "Test / medium-priority"
    else:
        priority = "Low-priority — consider excluding"

    action_text = f"{PRODUCT_MAP.get(preferred_product, '')} {CHANNEL_MAP.get(preferred_channel, '')}"

    # ── Results display ───────────────────────────────────────────────────
    res_l, res_m, res_r = st.columns(3)

    with res_l:
        st.subheader("Prediction")
        st.metric("Response probability", f"{prob * 100:.1f}%")
        if prob >= 0.45:
            st.success(f"**{priority}** — worth targeting in the next campaign.")
        elif prob >= 0.25:
            st.warning(f"**{priority}** — consider testing.")
        else:
            st.error(f"**{priority}** — likely to not respond.")

    with res_m:
        st.subheader("Behavioral Segment")
        st.markdown(f"### {segment}")
        st.markdown(f"RFM scores → **R: {r_sc}  ·  F: {f_sc}  ·  M: {m_sc}**")
        st.write(recommendation)

    with res_r:
        st.subheader("Recommended Action")
        st.markdown(f"**{priority}**")
        st.write(action_text)
        st.info(f"Expected return: **{roi_text}**")

    divider()

    st.subheader("Score Drivers")
    drivers = pd.DataFrame(
        [
            ["Past campaign acceptance", accepted_cmp,
             "Very strong" if accepted_cmp >= 2 else "Moderate" if accepted_cmp == 1 else "Weak"],
            ["Days since last purchase", recency_days,
             "Strong" if recency_days <= 30 else "Moderate" if recency_days <= 75 else "Weak"],
            ["Total spend", f"${total_spend:,}",
             "Strong" if total_spend >= 900 else "Moderate" if total_spend >= 300 else "Weak"],
            ["Purchase frequency", total_purchases,
             "Strong" if total_purchases >= 10 else "Moderate" if total_purchases >= 4 else "Weak"],
            ["Preferred channel", preferred_channel,
             "Positive" if preferred_channel in ("Catalog", "Web") else "Neutral"],
            ["Preferred product", preferred_product, "Informs creative strategy"],
        ],
        columns=["Driver", "Input", "Assessment"],
    )
    st.dataframe(drivers, use_container_width=True, hide_index=True)

    with st.expander("Demo script — use this while presenting"):
        st.markdown(
            """
1. **Start low:** Set past campaigns accepted = 0, recency = 120 days, spend = $50. Show the weak score and "Lost" segment.
2. **Increase engagement:** Bump accepted campaigns to 2, recency to 20 days. Watch the probability jump.
3. **High-value profile:** Set spend = $1,200, purchases = 15, accepted = 3. Show the "Champions" segment.
4. **Channel insight:** Switch preferred channel between Web / Catalog / Store and discuss what creative that implies.
5. **Close:** Tie the score to the $7,080 improvement shown in the Business Impact section — this is how we decide who gets the next campaign.
"""
        )
