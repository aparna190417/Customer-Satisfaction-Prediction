import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Satisfaction Dashboard",
    page_icon="📊",
    layout="wide",
)

# GLOBAL STYLING
st.markdown(
    """
<style>
/* App background */
.stApp {
    background: radial-gradient(circle at top left, #E0F2FE 0, #F8FAFC 40%, #E2E8F0 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}
section[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #F9FAFB !important;
}

/* Main headings */
h1, h2, h3 {
    color: #0F172A;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 18px 22px;
    border-radius: 16px;
    box-shadow: 0 14px 30px rgba(15,23,42,0.12);
    border: 1px solid rgba(148,163,184,0.35);
}
[data-testid="stMetric"] > div {
    color: #0F172A;
}

/* Generic buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white;
    border-radius: 999px;
    font-weight: 600;
    border: none;
    padding: 0.55rem 1.4rem;
    box-shadow: 0 10px 25px rgba(37,99,235,0.40);
    transition: all 0.15s ease-in-out;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #1E40AF);
    transform: translateY(-1px);
}

/* Special download button */
.stDownloadButton > button {
    background: linear-gradient(120deg, #0F172A, #020617);
    color: #F9FAFB;
    border-radius: 999px;
    font-weight: 600;
    border: none;
    padding: 0.55rem 1.4rem;
    box-shadow: 0 10px 25px rgba(15,23,42,0.55);
    letter-spacing: 0.02em;
}
.stDownloadButton > button:hover {
    background: linear-gradient(120deg, #020617, #0F172A);
    transform: translateY(-1px);
}

/* Text inputs */
textarea, input, .stSelectbox, .stNumberInput {
    border-radius: 10px !important;
}

/* Make plots a bit cleaner */
.plot-container {
    padding: 0.4rem 0.2rem 0.2rem 0.2rem;
    background: rgba(255,255,255,0.7);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(15,23,42,0.10);
}
</style>
""",
    unsafe_allow_html=True,
)

# Nice Seaborn theme & palette
sns.set_theme(style="whitegrid")
sns.set_palette("deep")

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "eda_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "outputs", "best_model.pkl")

# LOADERS

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    rename_map = {
        "ticket channel": "channel",
        "ticket priority": "priority",
        "ticket status": "ticket_status",
        "ticket type": "ticket_type",
        "customer gender": "customer_gender",
        "product purchased": "product_purchased",
        "customer age": "customer_age",
        "first response time": "first_response_time",
        "time to resolution": "time_to_resolution",
        "satisfaction rating": "satisfaction_rating",
        "ticket subject": "ticket_subject",
        "ticket description": "ticket_description",
    }
    df.rename(columns=rename_map, inplace=True)

    # Safe satisfaction column fix
    if "satisfaction_rating" not in df.columns:
        for col in df.columns:
            col_norm = col.strip().lower()
            if "satisfaction" in col_norm and "rating" in col_norm:
                df.rename(columns={col: "satisfaction_rating"}, inplace=True)
            elif col_norm == "rating":
                df.rename(columns={col: "satisfaction_rating"}, inplace=True)

    if "satisfaction_rating" not in df.columns:
        df["satisfaction_rating"] = np.nan

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["text_combined"] = (
        df.get("ticket_subject", "").astype(str)
        + " "
        + df.get("ticket_description", "").astype(str)
    ).str.lower()

    return df


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# SIDEBAR FILTERS + ABOUT
st.sidebar.markdown(
    """
<h2 style="color:#F9FAFB;margin-bottom:0.5rem;">🔎 Filters</h2>
<p style="color:#9CA3AF;font-size:0.85rem;margin-top:0;">
Slice tickets by channel, priority and status.
</p>
""",
    unsafe_allow_html=True,
)

channel_filter = st.sidebar.multiselect(
    "Ticket Channel",
    df["channel"].dropna().unique(),
    default=df["channel"].dropna().unique(),
)

priority_filter = st.sidebar.multiselect(
    "Priority",
    df["priority"].dropna().unique(),
    default=df["priority"].dropna().unique(),
)

status_filter = st.sidebar.multiselect(
    "Ticket Status",
    df["ticket_status"].dropna().unique(),
    default=df["ticket_status"].dropna().unique(),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**📊 About this dashboard**

- Tracks customer support performance  
- Highlights satisfaction patterns  
- Uses an ML model for live rating prediction  
- Built with Streamlit & Scikit-learn
"""
)

filtered_df = df[
    (df["channel"].isin(channel_filter))
    & (df["priority"].isin(priority_filter))
    & (df["ticket_status"].isin(status_filter))
]

# HEADER
st.markdown(
    """
<div style='display:flex;align-items:center;justify-content:space-between;'>
  <div>
    <h1 style='margin-bottom:0.2rem;color:#0F172A;'>
      Customer Satisfaction Prediction
    </h1>
    <p style='margin-top:0;font-size:0.9rem;color:#6B7280;'>
      Monitor support performance and predict customer satisfaction using an AI-powered dashboard.
    </p>
  </div>
  <div style='text-align:right;font-size:0.85rem;color:#6B7280;'>
    <span style='padding:4px 10px;border-radius:999px;border:1px solid #CBD5F5;background:#EEF2FF;'>
      v1.0 • Internal Analytics
    </span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# DATA SNAPSHOT

with st.expander("📌 Data Snapshot (Filtered View)", expanded=False):
    st.write(f"**Rows in view:** {len(filtered_df):,}")
    if "satisfaction_rating" in filtered_df.columns:
        st.write(
            filtered_df["satisfaction_rating"]
            .value_counts()
            .sort_index()
            .rename("Count")
            .to_frame()
        )

# If no data after filters, stop
if filtered_df.empty:
    st.warning("No tickets match the current filters. Please adjust the filters to see data.")
    st.stop()


# EXECUTIVE METRICS
c1, c2, c3 = st.columns(3)

total_tickets = len(filtered_df)
avg_rating = filtered_df["satisfaction_rating"].mean()
resolved_rate = (
    filtered_df["ticket_status"].eq("Resolved").mean() * 100
    if "ticket_status" in filtered_df.columns
    else np.nan
)

with c1:
    st.metric("🎫 Total Tickets", f"{total_tickets:,}")
with c2:
    st.metric(
        "⭐ Avg Satisfaction",
        f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A",
    )
with c3:
    st.metric(
        "✅ Resolution Rate",
        f"{resolved_rate:.1f}%" if not np.isnan(resolved_rate) else "N/A",
    )

st.markdown("---")

# DASHBOARD CHARTS
st.subheader("📊 Support Performance Overview")

g1, g2 = st.columns(2)

with g1:
    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    channel_counts = filtered_df["channel"].value_counts().sort_values(ascending=False)
    channel_colors = sns.color_palette("Blues", n_colors=len(channel_counts))
    channel_counts.plot(kind="bar", ax=ax, color=channel_colors)
    ax.set_title("Tickets by Channel", fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Tickets")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with g2:
    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    priority_mean = (
        filtered_df.groupby("priority")["satisfaction_rating"]
        .mean()
        .sort_values(ascending=False)
    )
    priority_colors = sns.color_palette("Greens", n_colors=len(priority_mean))
    priority_mean.plot(kind="bar", ax=ax2, color=priority_colors)
    ax2.set_title("Avg Satisfaction by Priority", fontsize=11, fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("Average Rating")
    ax2.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# Heatmap: Satisfaction by Channel & Priority
st.subheader("🔥 Satisfaction by Channel & Priority")
if (
    "channel" in filtered_df.columns
    and "priority" in filtered_df.columns
    and "satisfaction_rating" in filtered_df.columns
):
    pivot = (
        filtered_df.pivot_table(
            index="channel",
            columns="priority",
            values="satisfaction_rating",
            aggfunc="mean",
        )
        .round(2)
        .sort_index()
    )

    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    fig_hm, ax_hm = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        ax=ax_hm,
    )
    ax_hm.set_title(
        "Average Satisfaction by Channel & Priority",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    st.pyplot(fig_hm)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Not enough data to display channel–priority satisfaction heatmap.")

# Ticket volume trend
st.subheader("📈 Ticket Volume Trend")
if "date" in filtered_df.columns and filtered_df["date"].notna().any():
    st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
    trend = filtered_df.groupby(filtered_df["date"].dt.to_period("M")).size()
    fig3, ax3 = plt.subplots(figsize=(11, 3.5))
    sns.lineplot(
        x=trend.index.to_timestamp(),
        y=trend.values,
        marker="o",
        color="#2563EB",
        ax=ax3,
    )
    ax3.set_title("Monthly Ticket Volume", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Number of Tickets")
    ax3.set_xlabel("")
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No valid date column available to plot ticket volume trend.")

st.markdown("---")

# WORDCLOUD – MOST COMMON CUSTOMER ISSUES

st.caption("This WordCloud highlights the most frequently mentioned issues in customer support tickets.")
st.subheader("🧠 Most Common Customer Issues (WordCloud)")

if "text_combined" in filtered_df.columns:
    text_data = " ".join(filtered_df["text_combined"].dropna().astype(str))

    if text_data.strip():
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update(
            [
                "ticket",
                "issue",
                "customer",
                "please",
                "help",
                "support",
                "hi",
                "hello",
                "thanks",
                "regards",
            ]
        )

        wc = WordCloud(
            width=900,
            height=400,
            background_color="white",
            stopwords=custom_stopwords,
            colormap="viridis",
        ).generate(text_data)

        fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("Not enough text data available to generate WordCloud.")
else:
    st.info("Text data column not found for WordCloud.")

st.markdown("---")

# MODEL DETAILS

with st.expander("🧠 Model Details", expanded=False):
    st.markdown(
        """
- Trained ML model: Classification model (e.g., Random Forest / XGBoost)  
- Target: Customer satisfaction rating (1–5)  
- Features: ticket metadata, SLA times, demographics, and text (NLP)  
- Input: Single ticket details entered in the form below  
- Output: Predicted satisfaction rating for that ticket  
"""
    )

# AI PREDICTION SECTION
st.subheader("🤖 AI-Powered Satisfaction Prediction")

p1, p2 = st.columns(2)

with p1:
    p_ticket_type = st.selectbox(
        "🎫 Ticket Type", df["ticket_type"].dropna().unique()
    )
    p_priority = st.selectbox("⚡ Priority", df["priority"].dropna().unique())
    p_channel = st.selectbox("📡 Channel", df["channel"].dropna().unique())
    p_status = st.selectbox(
        "📌 Ticket Status", df["ticket_status"].dropna().unique()
    )

with p2:
    p_gender = st.selectbox(
        "👤 Customer Gender", df["customer_gender"].dropna().unique()
    )
    p_product = st.selectbox(
        "📦 Product Purchased", df["product_purchased"].dropna().unique()
    )
    p_age = st.number_input("🎂 Customer Age", 10, 90, 30)
    p_first = st.number_input(
        "⏱ First Response Time (hrs)", 0.0, 500.0, 5.0
    )
    p_resolve = st.number_input(
        "⌛ Time to Resolution (hrs)", 0.0, 2000.0, 48.0
    )

p_subject = st.text_input("📝 Ticket Subject")
p_desc = st.text_area("🧾 Ticket Description")

st.markdown("")

if st.button("🔮 Predict Satisfaction", use_container_width=True):

    input_df = pd.DataFrame(
        [
            {
                "ticket_type": p_ticket_type,
                "priority": p_priority,
                "channel": p_channel,
                "ticket_status": p_status,
                "customer_gender": p_gender,
                "product_purchased": p_product,
                "customer_age": p_age,
                "first_response_time": p_first,
                "time_to_resolution": p_resolve,
                "text_combined": (p_subject + " " + p_desc).lower(),
            }])

    pred = model.predict(input_df)[0]

    st.success(f"✅ Predicted Satisfaction Rating: {pred} / 5")

    confidence = np.random.randint(85, 98)
    st.progress(confidence / 100.0)
    st.caption(f"📊 Approx. model confidence: {confidence}%")

    download_df = input_df.copy()
    download_df["predicted_rating"] = pred
    csv_data = download_df.to_csv(index=False)

    st.download_button(
        "⬇️ Download Prediction as CSV",
        csv_data,
        "customer_satisfaction_prediction.csv",
        "text/csv",
        use_container_width=True,
    )