import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, norm, expon, t, chi2_contingency
import io

st.set_page_config(page_title="Statistics Dashboard by Group 3", layout="centered")
st.title("Statistics Dashboard")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Basic Statistics", "Probability Distributions", "Inferential Statistics",
    "Regression", "Visualization", "About"
])

with tab1:
    st.header("Basic Descriptive Statistics")
    data_str = st.text_input("Enter Data (comma-separated):", "5,7,8,5,9,10,6,6,7")
    if st.button("Calculate", key="basic"):
        try:
            data = np.array([float(x) for x in data_str.split(",")])
            mean = np.mean(data)
            median = np.median(data)
            counts = np.bincount(data.astype(int))
            mode = np.argmax(counts)
            variance = np.var(data)
            std = np.std(data)
            st.markdown(f"""
            - **Mean:** {mean:.2f}
            - **Median:** {median}
            - **Mode:** {mode}
            - **Variance:** {variance:.2f}
            - **Std Dev:** {std:.2f}
            """)
        except Exception as e:
            st.error("Please enter valid numbers.")

with tab2:
    st.header("Probability Distributions")
    dist_type = st.selectbox("Distribution", ["Binomial", "Poisson", "Normal", "Exponential"])
    chart_buf = None
    if dist_type == "Binomial":
        n = st.number_input("n (number of trials)", min_value=1, value=10)
        p = st.number_input("p (probability of success)", min_value=0.0, max_value=1.0, value=0.5)
        x = st.number_input("x (successes)", min_value=0, max_value=int(n), value=5)
        func = st.selectbox("Function", ["PMF (P(X = x))", "CDF (P(X ≤ x))"])
        if st.button("Calculate", key="binom"):
            if func.startswith("PMF"):
                val = binom.pmf(x, n, p)
                st.write(f"P(X = {x}) = {val:.5f}")
            else:
                val = binom.cdf(x, n, p)
                st.write(f"P(X ≤ {x}) = {val:.5f}")
            xs = np.arange(0, n+1)
            ys = binom.pmf(xs, n, p)
            fig, ax = plt.subplots()
            ax.bar(xs, ys)
            ax.set_xlabel("x")
            ax.set_ylabel("PMF")
            st.pyplot(fig)
            chart_buf = io.BytesIO()
            fig.savefig(chart_buf, format="png")
            chart_buf.seek(0)
            st.download_button(
                label="Download Chart as PNG",
                data=chart_buf,
                file_name="binomial_chart.png",
                mime="image/png"
            )
            plt.close(fig)
    elif dist_type == "Poisson":
        lam = st.number_input("λ (mean rate)", min_value=0.01, value=3.0)
        x = st.number_input("x (events)", min_value=0, value=2)
        func = st.selectbox("Function", ["PMF (P(X = x))", "CDF (P(X ≤ x))"])
        if st.button("Calculate", key="poisson"):
            if func.startswith("PMF"):
                val = poisson.pmf(x, lam)
                st.write(f"P(X = {x}) = {val:.5f}")
            else:
                val = poisson.cdf(x, lam)
                st.write(f"P(X ≤ {x}) = {val:.5f}")
            xs = np.arange(0, max(10, int(lam*2))+1)
            ys = poisson.pmf(xs, lam)
            fig, ax = plt.subplots()
            ax.bar(xs, ys)
            ax.set_xlabel("x")
            ax.set_ylabel("PMF")
            st.pyplot(fig)
            chart_buf = io.BytesIO()
            fig.savefig(chart_buf, format="png")
            chart_buf.seek(0)
            st.download_button(
                label="Download Chart as PNG",
                data=chart_buf,
                file_name="poisson_chart.png",
                mime="image/png"
            )
            plt.close(fig)
    elif dist_type == "Normal":
        mu = st.number_input("μ (mean)", value=0.0)
        sigma = st.number_input("σ (std dev)", min_value=0.01, value=1.0)
        x = st.number_input("x", value=1.0)
        func = st.selectbox("Function", ["PDF (f(x))", "CDF (P(X ≤ x))"])
        if st.button("Calculate", key="normal"):
            if func.startswith("PDF"):
                val = norm.pdf(x, mu, sigma)
                st.write(f"f({x}) = {val:.5f}")
            else:
                val = norm.cdf(x, mu, sigma)
                st.write(f"P(X ≤ {x}) = {val:.5f}")
            xs = np.linspace(mu-4*sigma, mu+4*sigma, 200)
            ys = norm.pdf(xs, mu, sigma)
            fig, ax = plt.subplots()
            ax.plot(xs, ys)
            ax.set_xlabel("x")
            ax.set_ylabel("PDF")
            st.pyplot(fig)
            chart_buf = io.BytesIO()
            fig.savefig(chart_buf, format="png")
            chart_buf.seek(0)
            st.download_button(
                label="Download Chart as PNG",
                data=chart_buf,
                file_name="normal_chart.png",
                mime="image/png"
            )
            plt.close(fig)
    elif dist_type == "Exponential":
        lam = st.number_input("λ (rate)", min_value=0.01, value=1.0)
        x = st.number_input("x", min_value=0.0, value=1.0)
        func = st.selectbox("Function", ["PDF (f(x))", "CDF (P(X ≤ x))"])
        if st.button("Calculate", key="exp"):
            if func.startswith("PDF"):
                val = expon.pdf(x, scale=1/lam)
                st.write(f"f({x}) = {val:.5f}")
            else:
                val = expon.cdf(x, scale=1/lam)
                st.write(f"P(X ≤ {x}) = {val:.5f}")
            xs = np.linspace(0, 5/lam, 200)
            ys = expon.pdf(xs, scale=1/lam)
            fig, ax = plt.subplots()
            ax.plot(xs, ys)
            ax.set_xlabel("x")
            ax.set_ylabel("PDF")
            st.pyplot(fig)
            chart_buf = io.BytesIO()
            fig.savefig(chart_buf, format="png")
            chart_buf.seek(0)
            st.download_button(
                label="Download Chart as PNG",
                data=chart_buf,
                file_name="exponential_chart.png",
                mime="image/png"
            )
            plt.close(fig)

with tab3:
    st.header("Inferential Statistics")
    st.subheader("Confidence Interval (Z)")
    mean = st.number_input("Sample Mean (x̄)", value=100.0)
    std = st.number_input("Sample Std Dev (s)", value=15.0)
    n = st.number_input("Sample Size (n)", min_value=1, value=30)
    level = st.number_input("Confidence Level (%)", min_value=50, max_value=99, value=95)
    if st.button("Calculate CI"):
        z = {90:1.645, 95:1.96, 99:2.576}.get(level, 1.96)
        moe = z * (std / np.sqrt(n))
        st.write(f"{level}% CI: [{mean-moe:.2f}, {mean+moe:.2f}] (±{moe:.2f})")

    st.subheader("One-Sample Hypothesis Test (Z)")
    mu0 = st.number_input("Population Mean (μ₀)", value=100.0)
    mean2 = st.number_input("Sample Mean (x̄)", value=103.0)
    std2 = st.number_input("Sample Std Dev (s)", value=10.0)
    n2 = st.number_input("Sample Size (n)", min_value=1, value=30, key="n2")
    alpha = st.number_input("Significance Level (α)", min_value=0.001, max_value=0.5, value=0.05)
    if st.button("Test", key="ztest"):
        se = std2 / np.sqrt(n2)
        zstat = (mean2 - mu0) / se
        pval = 2 * (1 - norm.cdf(abs(zstat)))
        st.write(f"Z-statistic: {zstat:.3f}, P-value: {pval:.4f}")
        st.write("Conclusion:", "Reject H₀" if pval < alpha else "Fail to reject H₀")

    st.subheader("Two-Sample t-Test")
    mean1 = st.number_input("Mean 1", value=105.0)
    sd1 = st.number_input("SD 1", value=15.0)
    n1 = st.number_input("n1", min_value=1, value=30, key="n1")
    mean2b = st.number_input("Mean 2", value=100.0)
    sd2 = st.number_input("SD 2", value=12.0)
    n2b = st.number_input("n2", min_value=1, value=30, key="n2b")
    if st.button("Test", key="ttest"):
        pooled_var = ((n1-1)*sd1**2 + (n2b-1)*sd2**2) / (n1+n2b-2)
        se = np.sqrt(pooled_var * (1/n1 + 1/n2b))
        tstat = (mean1 - mean2b) / se
        df = n1 + n2b - 2
        st.write(f"t-statistic: {tstat:.3f}, df: {df}")

    st.subheader("Chi-Square Test for Independence")
    table_str = st.text_area("Enter 2D array (e.g., [[90,60],[30,20]])", "[[90,60],[30,20]]")
    if st.button("Calculate Chi2"):
        try:
            table = np.array(eval(table_str))
            chi2, p, dof, expected = chi2_contingency(table)
            st.write(f"Chi2: {chi2:.3f}, df: {dof}, p-value: {p:.4f}")
        except Exception as e:
            st.error("Invalid table format.")

with tab4:
    st.header("Linear Regression & Pearson Correlation")
    x_str = st.text_input("X values (comma-separated):", "1,2,3,4,5")
    y_str = st.text_input("Y values (comma-separated):", "2,4,5,4,5")
    if st.button("Calculate Regression"):
        try:
            x = np.array([float(i) for i in x_str.split(",")])
            y = np.array([float(i) for i in y_str.split(",")])
            if len(x) != len(y):
                st.error("X and Y must have the same length.")
            else:
                slope, intercept = np.polyfit(x, y, 1)
                r = np.corrcoef(x, y)[0,1]
                st.write(f"y = {slope:.2f}x + {intercept:.2f}")
                st.write(f"Pearson r = {r:.3f}")
                fig, ax = plt.subplots()
                ax.scatter(x, y, label="Data")
                ax.plot(x, slope*x + intercept, color="red", label="Regression Line")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.legend()
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button(
                    label="Download Chart as PNG",
                    data=buf,
                    file_name="regression_chart.png",
                    mime="image/png"
                )
                plt.close(fig)
        except Exception as e:
            st.error("Please enter valid numbers.")

with tab5:
    st.header("Dynamic Charts")
    chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Pie", "Scatter"])
    labels = st.text_input("Labels (comma):", "A,B,C")
    data = st.text_input("Data (comma):", "10,20,30")
    if st.button("Render Chart"):
        try:
            labels_list = [l.strip() for l in labels.split(",")]
            data_list = [float(d) for d in data.split(",")]
            fig, ax = plt.subplots()
            if chart_type == "Bar":
                ax.bar(labels_list, data_list)
            elif chart_type == "Line":
                ax.plot(labels_list, data_list, marker="o")
            elif chart_type == "Pie":
                ax.pie(data_list, labels=labels_list, autopct="%1.1f%%")
            elif chart_type == "Scatter":
                ax.scatter(labels_list, data_list)
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="Download Chart as PNG",
                data=buf,
                file_name="dynamic_chart.png",
                mime="image/png"
            )
            plt.close(fig)
        except Exception as e:
            st.error("Please enter valid labels and numbers.")

    st.header("Histogram & Box Plot")
    hist_data = st.text_input("Data (comma):", "5,7,8,5,9,10,6,6,7")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Histogram"):
            try:
                arr = np.array([float(i) for i in hist_data.split(",")])
                fig, ax = plt.subplots()
                ax.hist(arr, bins="auto", color="#60a5fa")
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button(
                    label="Download Histogram as PNG",
                    data=buf,
                    file_name="histogram.png",
                    mime="image/png"
                )
                plt.close(fig)
            except Exception as e:
                st.error("Please enter valid numbers.")
    with col2:
        if st.button("Box Plot"):
            try:
                arr = np.array([float(i) for i in hist_data.split(",")])
                fig, ax = plt.subplots()
                ax.boxplot(arr, vert=True)
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button(
                    label="Download Box Plot as PNG",
                    data=buf,
                    file_name="boxplot.png",
                    mime="image/png"
                )
                plt.close(fig)
            except Exception as e:
                st.error("Please enter valid numbers.")

with tab6:
    st.write("This dashboard helps you explore fundamental and advanced topics in statistics, from confidence intervals to regression and visualization tools.")
    st.write("Group 3")
    st.write("Group Members:")
    st.write("1. Cut Kheysa Sakbania")
    st.write("2. Clarizza Revalentina Setiawan")
    st.write("3. Elmira Jacinda Wahid")
    