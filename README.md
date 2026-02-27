# iFood Marketing Analytics Case Study

> **Customer Segmentation & Campaign Optimization for a Food Delivery Company**

A comprehensive analysis of 2,205 customers using clustering, RFM analysis, and predictive modeling to optimize marketing campaigns and increase profitability.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📊 Business Impact

**Current State:** Untargeted campaigns losing $5,400 per cycle (15% response rate, 2,205 contacts)

**Optimized State:** Targeted campaigns generating $1,680 profit (46.5% response rate, 127 contacts)

**Net Improvement:** **$7,080 per campaign** | **132% ROI**

---

## 🎯 Executive Summary

### The Challenge
iFood needed to optimize marketing spend and improve campaign response rates. Previous campaigns had:
- Low response rates (15%)
- High customer acquisition costs
- Unclear targeting strategy
- No differentiation between customer segments

### The Solution
Developed a data-driven segmentation and targeting framework using:
1. **Customer Clustering** (K-Means) - Identified 4 demographic profiles
2. **RFM Analysis** - Created 9 behavioral segments
3. **Predictive Modeling** (Logistic Regression) - 91% ROC-AUC score
4. **Channel & Product Analysis** - Matched segments to preferences

### Key Findings

#### 🏆 **Champions Segment** (275 customers)
- **Profile:** Empty Nesters, $1,150+ average spend
- **Response Rate:** 29.8% (2x average)
- **Channel:** Catalog-first (30% of purchases)
- **Products:** Wine (50% of spend), Meat (20%)
- **Action:** Premium wine catalogs, VIP experiences
- **Expected ROI:** $50-60 revenue per $10 investment

#### 📈 **Potential Loyalists** (378 customers)
- **Profile:** Recent, frequent buyers, $424 average spend
- **Response Rate:** 24.3%
- **Channel:** Web-dominant (37% of purchases)
- **Products:** Entry-level wine, meat bundles
- **Action:** Conversion campaigns to increase spending
- **Expected ROI:** $25-30 revenue per $10 investment

#### 💎 **Loyal Customers** (436 customers)
- **Profile:** Established relationship, $1,180+ spend
- **Response Rate:** 16.7%
- **Channel:** Omnichannel (balanced across web/catalog/store)
- **Products:** Wine + Meat enthusiasts
- **Action:** Loyalty rewards, upgrade to Champions
- **Expected ROI:** $30-35 revenue per $10 investment

### Strategic Recommendations

#### 1. **Immediate Actions** (Next Campaign)
**Target:** Champions + Potential Loyalists + Top 150 Loyal Customers (800 total)
- **Theme:** Premium Wine Selection
- **Channels:** 50% Catalog, 30% Email/Web, 20% Retargeting
- **Expected Results:** 180-200 conversions, $2,000-4,000 profit

#### 2. **Product Expansion Opportunities**
- **Wine Dominance:** 45-55% of high-value segment spending
  - Introduce premium tiers ($75-150 bottles)
  - Create wine subscription program
- **Underserved Categories:** Fruits (9%), Fish (7%), Sweets (5%)
  - Bundle with primary purchases (Wine + Cheese + Fruit)
  - Recipe cards and pairing guides in catalogs

#### 3. **Channel Optimization**
- **Digital Growth:** 442 "Young Families" are store-heavy
  - Mobile app with in-store pickup
  - QR code campaigns linking store to web
- **Catalog Excellence:** High-value customers over-index on catalog
  - Premium design for Champions
  - Seasonal lookbooks

#### 4. **Segment Migration Paths**
- **Potential Loyalists → Loyal Customers**
  - 40% conversion target (150 customers)
  - $113,000 incremental revenue opportunity
- **Loyal Customers → Champions**
  - Reduce purchase recency from 70 to <30 days
  - 20% conversion target (87 customers)

#### 5. **Cost Reduction**
- **Stop Targeting:** Lost, Hibernating, Needs Attention (638 customers)
  - <5% response rate
  - Draining campaign budgets
- **Quarterly Win-Back Only:** Aggressive discounts via email
- **Annual Sunset:** Remove non-responders after 3 failed attempts

---

## 📁 Project Structure

```
marketing-analytics-case-study/
│
├── data/
│   ├── raw/
│   │   ├── ifood_df.csv              # Original dataset (2,240 customers)
│   │   └── dictionary.csv            # Data dictionary
│   └── processed/
│       ├── ifood_df_cleaned.csv      # Cleaned (2,205 customers)
│       ├── ifood_df_with_clusters.csv # + K-Means clusters
│       ├── ifood_df_with_rfm.csv     # + RFM segments
│       └── dictionary.csv            # Updated dictionary
│
├── notebooks/
│   ├── 01-Exploration.ipynb          # EDA & cleaning
│   ├── 02-Clustering.ipynb           # K-Means segmentation
│   ├── 03-RFM-Analysis.ipynb         # RFM scoring & segments
│   ├── 04-Campaign-Response-Prediction.ipynb  # ML models
│   └── 05-Channel-Spend-Insights.ipynb        # Actionable recommendations
│
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

---

## 🔍 Methodology

### Phase 1: Data Exploration & Cleaning
- **Dataset:** 2,240 customers, 38 features
- **Cleaning:** Removed outliers (income >$600K, age >90), imputed missing values
- **Final:** 2,205 customers (98.4% retention)
- **Key Features:** Demographics, spending, channel usage, campaign history

### Phase 2: Customer Clustering (K-Means)
- **Optimal K:** 4 clusters (Elbow method + Silhouette score)
- **Features:** Income, Age, TotalChildren, MntTotal (standardized)
- **Result:** 4 demographic segments
  1. **Empty Nesters** (519) - High income, no kids, $1,170 spend
  2. **Mature Families w/ Teens** (672) - Moderate income, teens, $656 spend
  3. **Stretched Parents** (406) - Low income, young kids, $197 spend
  4. **Young Families on Budget** (608) - Low income, young kids, $118 spend

### Phase 3: RFM Analysis
- **R** (Recency): Days since last purchase → 5 quintiles
- **F** (Frequency): Total # purchases → 5 quintiles
- **M** (Monetary): Total spending → 5 quintiles
- **Segments:** 9 behavioral groups (Champions to Lost)

**Segment Definitions:**
| Segment | R | F | M | Description |
|---------|---|---|---|-------------|
| Champions | 4-5 | 4-5 | 4-5 | Best customers: recent, frequent, high-spend |
| Loyal Customers | 1-2 | 4-5 | 4-5 | Regular buyers but haven't purchased recently |
| Potential Loyalists | 4-5 | 2-3 | 2-3 | Recent, growing frequency & spend |
| Promising | 3-4 | 3-4 | 3-4 | Moderate on all dimensions |
| At Risk | 1-2 | 3-4 | 3-4 | Good customers slipping away |
| Recent Customers | 4-5 | 1 | 1 | New, low activity |
| Needs Attention | 3 | 1-2 | 1-2 | Below average, need engagement |
| Hibernating | 1-2 | 2-3 | 2-3 | Low recent activity |
| Lost | 1-2 | 1 | 1 | Minimal engagement, likely churned |

### Phase 4: Predictive Modeling
**Goal:** Predict campaign response to optimize targeting

**Models Tested:**
1. **Logistic Regression** ⭐ *Selected*
   - ROC-AUC: 0.908
   - Precision: 0.49 | Recall: 0.84
   - **Why chosen:** Best profit ($1,680), interpretability, fast

2. **Random Forest**
   - ROC-AUC: 0.893
   - Precision: 0.50 | Recall: 0.61

3. **XGBoost**
   - ROC-AUC: 0.904
   - Precision: 0.58 | Recall: 0.52

**Key Predictors:**
1. **AcceptedCmpOverall** (past campaigns) - Strongest by far
2. **Customer_Days** (tenure)
3. **M_Score** (monetary value)
4. **NumCatalogPurchases** (channel preference)

**Threshold Optimization:**
- Default (0.5): Not profit-optimal
- Optimal (0.45): Maximizes profit at $1,680
- Targets 127 customers with 46.5% expected conversion

### Phase 5: Channel & Product Analysis
**Channel Preferences:**
- **High-Value:** Multi-channel, catalog-heavy (25-30%)
- **At-Risk:** Store-dominant (40-45%)
- **Lost/Hibernating:** Low activity across all channels

**Product Insights:**
- **Wine:** Dominates high-value segments (45-55% of spend)
- **Meat:** Second priority (20-25%)
- **Growth Categories:** Fruits, Fish, Sweets (<10% each)
- **Gold Products:** Only Champions buy (15-20%)

**Campaign Performance:**
- Campaigns 3 & 4 performed best for Champions (30-35% response)
- Last Campaign works well for Potential Loyalists (24%)
- Lost/Hibernating: <5% response to any campaign

---

## 🛠️ Technologies Used

**Languages & Tools:**
- Python 3.8+
- Jupyter Notebook

**Libraries:**
- **Data Manipulation:** pandas, numpy
- **Visualization:** plotly, seaborn, matplotlib
- **Machine Learning:** scikit-learn, xgboost
- **Clustering:** K-Means, PCA
- **Statistical Analysis:** scipy, statsmodels

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
pip or conda package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/marketing-analytics-case-study.git
cd marketing-analytics-case-study
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter**
```bash
jupyter notebook
```

5. **Run notebooks in order:**
   - Start with `01-Exploration.ipynb`
   - Follow through `02`, `03`, `04`, `05`

### Quick Start (View Only)
Notebooks include all outputs - no need to run code to see results.

---

## 📈 Key Metrics & Results

### Model Performance
| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|---------------------|---------------|---------|
| ROC-AUC | **0.908** ⭐ | 0.893 | 0.904 |
| Precision | 0.49 | 0.50 | **0.58** |
| Recall | **0.84** | 0.61 | 0.52 |
| F1-Score | 0.62 | 0.55 | 0.55 |
| False Positives | 59 | **41** | **25** |
| False Negatives | **11** | 26 | 32 |

### Business Metrics
| Metric | Current (Untargeted) | Optimized (Model) | Improvement |
|--------|----------------------|-------------------|-------------|
| Customers Contacted | 2,205 | 127 | -94.2% |
| Response Rate | 15.1% | 46.5% | +208% |
| Expected Conversions | 333 | 59 | -82.3% |
| Campaign Cost | $22,050 | $1,270 | -94.2% |
| Revenue | $16,650 | $2,950 | -82.3% |
| **Profit** | **-$5,400** | **$1,680** | **$7,080** |
| **ROI** | -24.5% | **132%** | **+157%** |

### Segment Performance
| Segment | Size | Response Rate | Avg Spend | Recommended Action |
|---------|------|---------------|-----------|-------------------|
| Champions | 275 | 29.8% | $1,152 | Target all campaigns ✅ |
| Potential Loyalists | 378 | 24.3% | $424 | Growth campaigns ✅ |
| Loyal Customers | 436 | 16.7% | $1,187 | Loyalty rewards ✅ |
| At Risk | 135 | 12.6% | $821 | Win-back campaigns ⚠️ |
| Promising | 104 | 12.5% | $619 | Seasonal testing ⚠️ |
| Recent Customers | 239 | 12.1% | $34 | Onboarding ⚠️ |
| Needs Attention | 106 | 7.5% | $98 | Exclude ❌ |
| Hibernating | 243 | 4.1% | $175 | Exclude ❌ |
| Lost | 289 | 3.1% | $35 | Exclude ❌ |

---

## 💡 Business Insights

### Finding #1: Past Behavior Predicts Future Response
**Insight:** Customers who accepted previous campaigns are 10x more likely to respond to the next one.

**Evidence:**
- 0 past campaigns → 8% response rate
- 4 past campaigns → 91% response rate

**Action:**
- Prioritize customers with AcceptedCmpOverall ≥ 1
- Create "engaged customer" fast track
- Don't waste budget on never-responders

### Finding #2: Wine Drives High-Value Segments
**Insight:** Wine represents 45-55% of spending for Champions and Loyal Customers.

**Evidence:**
- Champions: $629 avg wine spend (51% of total)
- Over-index by 300% vs overall average
- Strongly correlated with catalog usage

**Action:**
- Lead campaigns with premium wine offerings
- Create wine subscription program
- Bundle wine with complementary products (meat, cheese)

### Finding #3: Channel Preference Varies by Segment
**Insight:** Different segments have distinct channel behaviors.

**Evidence:**
- Empty Nesters: 30% catalog, 27% web, 43% store
- Young Families: 15% catalog, 20% web, 65% store
- Champions: Balanced multi-channel usage

**Action:**
- Segment-specific channel mix
- Premium catalogs for high-value customers
- Digital-first for younger segments

### Finding #4: Product Category Opportunities
**Insight:** Fruits, Fish, and Sweets are massively underserved (<10% of spend).

**Evidence:**
- All segments under-index on these categories
- Even Champions only spend 8% on fruits
- Cross-sell potential is untapped

**Action:**
- Bundling strategy: Wine + Cheese + Fruit
- Recipe cards in catalogs
- Sampling programs for new categories

### Finding #5: Deal Sensitivity Inverse to Value
**Insight:** High-value customers rarely buy on deals; low-value customers are deal-dependent.

**Evidence:**
- Champions: 8% deal purchases, $1,152 avg spend
- Recent Customers: 28% deal purchases, $34 avg spend
- Negative correlation between deal ratio and lifetime value

**Action:**
- Avoid discounting Champions (trains deal-seeking)
- Use deals strategically for at-risk reactivation
- Quality/exclusivity messaging for premium segments

---

## 🎓 Lessons Learned

### What Worked Well
1. **Multi-Phase Approach:** Combining clustering + RFM + ML provided comprehensive view
2. **Business-First Metrics:** Optimizing for profit, not just accuracy
3. **Threshold Optimization:** Found that default 0.5 threshold was suboptimal
4. **Visual Storytelling:** Plotly dashboards made insights accessible

### What I'd Do Differently
1. **Customer Lifetime Value:** Incorporate CLV predictions for long-term optimization
2. **Time Series Analysis:** Analyze seasonal patterns in purchasing behavior
3. **A/B Testing Framework:** Build simulation framework for campaign testing
4. **Real-Time Scoring:** Deploy model as API for real-time customer scoring

### Technical Challenges
1. **Class Imbalance:** 85/15 split required careful handling (balanced weights, threshold tuning)
2. **Feature Engineering:** Creating meaningful RFM segments required domain knowledge
3. **Interpretability vs Performance:** Chose simpler model (LR) over complex (XGB) for business adoption

---

## 📚 Further Reading

**Recommended Resources:**
- [RFM Analysis Best Practices](https://www.putler.com/rfm-analysis/)
- [Customer Segmentation Guide](https://www.optimove.com/resources/learning-center/customer-segmentation)
- [Predictive Marketing Analytics](https://www.marketingevolution.com/marketing-essentials/predictive-analytics)

**Related Projects:**
- [Customer Churn Prediction](https://github.com/topics/churn-prediction)
- [Market Basket Analysis](https://github.com/topics/market-basket-analysis)
- [Customer Lifetime Value](https://github.com/topics/customer-lifetime-value)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**Areas for Contribution:**
- Additional visualizations
- Advanced modeling techniques (neural networks, ensemble methods)
- Time series forecasting
- Interactive dashboard (Streamlit/Dash)
- Deployment pipeline

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **iFood** for providing the dataset via Kaggle
- **Kaggle** community for inspiration
- **Scikit-learn & XGBoost** teams for excellent ML libraries
- **Plotly** for interactive visualization tools

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.

---

**Data Source:** [Kaggle - Marketing Data](https://www.kaggle.com/datasets/jackdaoud/marketing-data)

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Built with Python, Jupyter, and data-driven decision making**

</div>
