import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Page Config

st.set_page_config(
    page_title="E-Commerce Sales Dashboard",
    page_icon="📊",
    layout="wide",
)

# Data Loading

@st.cache_data
def load_data():
    df = pd.read_csv("data/ecommerce_sales_analysis.csv")
    month_cols = [c for c in df.columns if c.startswith("sales_month_")]
    long_df = df.melt(
        id_vars=["product_id", "product_name", "category", "price",
                 "review_score", "review_count"],
        value_vars=month_cols,
        var_name="month",
        value_name="units_sold",
    )
    long_df["revenue"] = long_df["units_sold"] * long_df["price"]
    return df, long_df

df, long_df = load_data()

# Side bar 

st.sidebar.header("Filters")

categories = sorted(df["category"].unique())
selected_categories = st.sidebar.multiselect(
    "Category", categories, default=categories
)

price_min, price_max = int(df["price"].min()), int(df["price"].max())
price_range = st.sidebar.slider(
    "Price range ($)",
    min_value=0,
    max_value=price_max,
    value=(0, price_max),
)

rating_range = st.sidebar.slider(
    "Review Score",
    min_value=1.0,
    max_value=5.0,
    value=(1.0, 5.0),
    step=0.1,
)

filtered = df[
    df['category'].isin(selected_categories)
    & df["price"].between(*price_range)
    & df["review_score"].between(*rating_range)
]

filtered_long = long_df[long_df["product_id"].isin(filtered["product_id"])]

# Header
st.title("📊 E-Commerce Sales Dashboard")
st.caption("Interactive exploration of 1000 products across 7 categories")

if filtered.empty:
    st.warning("No product filters selected!")
    st.stop()

# KPI Row

col1, col2, col3, col4, col5 = st.columns(5)

total_units = filtered_long["units_sold"].sum()
total_revenue = filtered_long["revenue"].sum()
avg_rating = filtered["review_score"].mean()
n_products = filtered["product_id"].nunique()
n_categories = filtered["category"].nunique()

col1.metric("Products", f"{n_products:,}")
col2.metric("Categories", n_categories)
col3.metric("Total Units Sold", f"{total_units:,}")
col4.metric("Total Revenue", f"${total_revenue:,.0f}")
col5.metric("Avg Rating", f"{avg_rating:.2f}")

st.divider()

# Tab views

tab1, tab2, tab3, tab4 = st.tabs(
    ["📦 Category Explorer", "💵 Price vs. Sales", "📈 Monthly Trends", "🔍 Product Lookup"]
)

# Tab 1: Category Explorer

with tab1:
    st.subheader("Category Performance")

    cat_summary = (
        filtered_long.groupby("category")
        .agg(
            total_units=("units_sold", "sum"),
            total_revenue=("revenue", "sum"),
        )
        .join(
            filtered.groupby("category")["review_score"].mean().rename("avg_rating")
        )
        .sort_values("total_revenue", ascending=False)
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Revenue by category**")
        st.bar_chart(cat_summary["total_revenue"])

    with col_b:
        st.markdown("**Units sold by category**")
        st.bar_chart(cat_summary["total_units"])

    st.markdown("**Category summary table**")
    st.dataframe(
        cat_summary.style.format(
            {"total_units": "{:,.0f}", "total_revenue": "${:,.0f}", "avg_rating": "{:.2f}"}
        )
    )

    # Tab 2: Price vs. Sales

with tab2:
    st.subheader("Price vs. total annual sales")

    product_totals = (
        filtered_long.groupby(["product_id", "category"])
        .agg(
            units=("units_sold", "sum"),
            revenue=("revenue", "sum"),
        )
        .reset_index()
        .merge(filtered[["product_id", "price", "review_score"]], on="product_id")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=product_totals, x="price", y="units",
        hue="category", alpha=0.6, ax=ax,
    )
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Units sold (12 months)")
    ax.set_title("Price vs. Sales Volume")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    corr = product_totals["price"].corr(product_totals["units"])
    st.caption(f"Pearson correlation (price vs. units): **{corr:.3f}**")

# Tab3: Monthly Trenss
with tab3:
    st.subheader("Monthly sales trends")

    view_mode = st.radio(
        "View as", ["Total units", "Total revenue"], horizontal=True
    )

    metric = "units_sold" if view_mode == "Total units" else "revenue"

    monthly = (
        filtered_long.groupby(["month", "category"])[metric]
        .sum()
        .unstack("category")
    )
    monthly = monthly.reindex([f"sales_month_{i}" for i in range(1, 13)])
    monthly.index = [f"M{i}" for i in range(1, 13)]

    st.line_chart(monthly)

    st.markdown("**Overall monthly totals**")
    st.bar_chart(monthly.sum(axis=1))

# Tab 4: Product Lookup

with tab4:
    st.subheader("Product lookup")

    product_name = st.selectbox(
        "Choose a product", sorted(filtered["product_name"].unique())
    )

    row = filtered[filtered["product_name"] == product_name].iloc[0]
    product_history = long_df[long_df["product_id"] == row["product_id"]].copy()
    product_history = product_history.sort_values("month")
    product_history["month_label"] = [
        f"M{i}" for i in range(1, 13)
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Category", row["category"])
    c2.metric("Price", f"${row['price']:.2f}")
    c3.metric("Rating", f"{row['review_score']:.1f}")
    c4.metric("Reviews", f"{row['review_count']:,}")

    st.markdown("**12-month sales history**")
    st.bar_chart(product_history.set_index("month_label")["units_sold"])
