
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import re

# ==============================================================================
# 🚀 SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Enterprise Campaign Strategy Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { color: #002551; font-weight: 800; font-size: 28px; margin-bottom: 5px; }
    .sub-header { color: #475569; font-size: 14px; margin-bottom: 25px; }
    .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; text-align: center; }
    .metric-val { color: #0054B7; font-size: 24px; font-weight: bold; }
    .metric-lbl { color: #64748b; font-size: 12px; font-weight: bold; }
    .insight-box { background-color: #f0fdf4; border-left: 5px solid #16a34a; padding: 20px; border-radius: 5px; margin-bottom: 25px; }
    .insight-title { color: #166534; font-weight: 800; font-size: 18px; margin-bottom: 10px; }
    .insight-text { color: #1f2937; font-size: 15px; line-height: 1.5; margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧹 ARMORED AUTO-SCRUBBER ENGINE
# ==============================================================================
def clean_bilingual_suffix(name_str):
    if pd.isna(name_str): return "Unnamed Asset"
    return re.sub(r'(?i)[-_ ]+(FR|EN)\b', '', str(name_str)).strip()

def scrub_and_load_excel(uploaded_file):
    if uploaded_file is None: return None, None, None
    try:
        file_bytes = uploaded_file.read()
        is_csv = uploaded_file.name.lower().endswith('.csv')
        df_raw = pd.read_csv(io.BytesIO(file_bytes), header=None, low_memory=False) if is_csv else pd.read_excel(io.BytesIO(file_bytes), header=None)
            
        header_idx = 0
        for i in range(min(30, len(df_raw))):
            row_vals = [str(x).lower().strip() for x in df_raw.iloc[i].values]
            if 'merchandise name' in row_vals or 'total item views' in row_vals or 'sku' in row_vals:
                header_idx = i
                break
                
        df_clean = pd.read_csv(io.BytesIO(file_bytes), skiprows=header_idx, low_memory=False) if is_csv else pd.read_excel(io.BytesIO(file_bytes), skiprows=header_idx)
        df_clean.columns = [str(c).strip() for c in df_clean.columns]

        def get_col(exact_names):
            for exact in exact_names:
                for col in df_clean.columns:
                    if exact.lower() == col.lower(): return col
            for exact in exact_names:
                for col in df_clean.columns:
                    if exact.lower() in col.lower(): return col
            return None

        mapping = {
            'sku': get_col(['SKU', 'Merchandise ID']), 'name': get_col(['Merchandise Name', 'Name']),
            'display_type': get_col(['Display Type']), 'page': get_col(['Page Position', 'Page']),
            'brand': get_col(['Brand', 'Manufacturer']), 'orig_price': get_col(['Total Original Price', 'Original Price']),
            'curr_price': get_col(['Total Current Price', 'Current Price']),
            'c1': get_col(['Custom ID 1']), 'c2': get_col(['Custom ID 2']), 'c3': get_col(['Custom ID 3']), 
            'c4': get_col(['Custom ID 4']), 'c5': get_col(['Custom ID 5']), 'c6': get_col(['Custom ID 6']),
            'ret_cat': get_col(['Retailer Category']), 'goo_cat': get_col(['Google Category L1']),
            'views': get_col(['Total Item Views', 'Views']), 'clicks': get_col(['Total Item Clicks', 'Clicks']),
            'clips': get_col(['Total Clippings', 'Clips']), 'ttms': get_col(['Total Transfer to Merchant (TTMs)', 'Total Transfer to Merchant', 'TTMS'])
        }
        return df_clean, mapping, header_idx
    except Exception as e:
        st.error(f"Error scrubbing file setup: {str(e)}")
        return None, None, None

def process_metrics(df, m):
    def normalize_sku(val):
        s = str(val).strip()
        return s[:-2] if s.endswith('.0') else (s if s not in ['nan', 'NaN', 'None', ''] else "UNKNOWN")
        
    df['SKU'] = df[m['sku']].apply(normalize_sku) if m['sku'] else "UNKNOWN"
    df['Name'] = df[m['name']].astype(str).str.strip().apply(clean_bilingual_suffix) if m['name'] else "Unnamed Asset"
    df['Display_Type'] = df[m['display_type']].astype(str).str.upper().str.strip() if m['display_type'] else "PRODUCT"
    df['Page'] = df[m['page']].astype(str).str.extract(r'(\d+)').fillna(1).astype(int) if m['page'] else 1
    
    df['Brand'] = df[m['brand']].astype(str).str.strip() if m['brand'] and m['brand'] in df.columns else "UNKNOWN"
    is_sku_clone = (df['Brand'] == df['SKU']) | df['Brand'].isin(['nan', 'NaN', 'None', '', 'UNKNOWN'])
    df.loc[is_sku_clone, 'Brand'] = df.loc[is_sku_clone, 'Name'].apply(lambda x: str(x).split()[0].upper() if str(x).strip() != "" else "GENERIC")
        
    def safe_numeric(col_name):
        if m[col_name] and m[col_name] in df.columns:
            cleaned = df[m[col_name]].astype(str).str.replace(r'[^\d.]', '', regex=True).replace('', '0')
            return pd.to_numeric(cleaned, errors='coerce').fillna(0)
        return 0

    df['Views'], df['Clicks'], df['Clips'], df['TTMs'] = safe_numeric('views'), safe_numeric('clicks'), safe_numeric('clips'), safe_numeric('ttms')
    df['Orig_Price'], df['Curr_Price'] = safe_numeric('orig_price'), safe_numeric('curr_price')
    df['Discount_Pct'] = np.where(df['Orig_Price'] > 0, ((df['Orig_Price'] - df['Curr_Price']) / df['Orig_Price']) * 100, 0.0)
    df['Discount_Pct'] = np.where(df['Discount_Pct'] < 0, 0.0, df['Discount_Pct'])

    def run_waterfall(row):
        for key in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'ret_cat', 'goo_cat']:
            if m[key] and pd.notna(row[m[key]]):
                val = str(row[m[key]]).strip()
                if val not in ["", "NULL", "nan", "NaN", "None"]: return val
        return "General Merchandise"
    df['Category'] = df.apply(run_waterfall, axis=1)
    
    global_totals = {'views': df['Views'].sum(), 'clicks': df['Clicks'].sum(), 'clips': df['Clips'].sum(), 'ttms': df['TTMs'].sum()}
    is_invalid_sku = df['SKU'].isin(['UNKNOWN', '0', 'none', 'null', 'nan'])
    return df[~is_invalid_sku].copy(), df[is_invalid_sku & ((df['Display_Type'] == "LINK") | (df['Name'].str.contains('BANNER', case=False, na=False)))].copy(), global_totals

def extract_exact_metadata(df_clean):
    try:
        merchant = df_clean['Merchant Name'].dropna().iloc[0] if 'Merchant Name' in df_clean.columns else "Bumper to Bumper"
        run_name = df_clean['Flyer Run Name'].dropna().iloc[0] if 'Flyer Run Name' in df_clean.columns else "Active Flight"
        run_id = df_clean['Flyer Run ID'].dropna().iloc[0] if 'Flyer Run ID' in df_clean.columns else "N/A"
        date_from = str(df_clean['Daily Available From'].dropna().iloc[0]).split()[0] if 'Daily Available From' in df_clean.columns else "N/A"
        date_to = str(df_clean['Daily Available To'].dropna().iloc[0]).split()[0] if 'Daily Available To' in df_clean.columns else "N/A"
        return merchant, run_name, str(run_id)[:-2] if str(run_id).endswith('.0') else str(run_id), date_from, date_to
    except:
        return "Bumper to Bumper", "Active Flight", "N/A", "N/A", "N/A"

def process_scroll_file(scroll_file, period_name=None):
    file_bytes = scroll_file.read()
    is_csv = scroll_file.name.lower().endswith('.csv')
    df_raw = pd.read_csv(io.BytesIO(file_bytes), header=None, low_memory=False) if is_csv else pd.read_excel(io.BytesIO(file_bytes), header=None)
        
    header_idx = 0
    for i in range(min(20, len(df_raw))):
        if any(keyword in [str(x).lower().strip() for x in df_raw.iloc[i].values] for keyword in ['scroll depth', 'retention', 'readers', 'milestone', 'percentage']):
            header_idx = i
            break
            
    df_sc = pd.read_csv(io.BytesIO(file_bytes), skiprows=header_idx, low_memory=False) if is_csv else pd.read_excel(io.BytesIO(file_bytes), skiprows=header_idx)
    df_sc.columns = [str(c).strip() for c in df_sc.columns]
    cols_lower = [c.lower() for c in df_sc.columns]
    
    if 'scroll depth' in cols_lower and 'cumulative readers' in cols_lower and 'total readers' in cols_lower:
        get_exact = lambda name: next((c for c in df_sc.columns if c.lower() == name), None)
        sd_col, pr_col, cr_col, tr_col = get_exact('scroll depth'), get_exact('pages read'), get_exact('cumulative readers'), get_exact('total readers')
        
        if pr_col: df_sc[pr_col] = pd.to_numeric(df_sc[pr_col], errors='coerce').fillna(0)
        agg = df_sc.groupby(sd_col).agg({pr_col: 'mean' if pr_col else 'first', cr_col: 'sum', tr_col: 'sum'}).reset_index()
        agg['Retention'] = np.where(agg[tr_col] > 0, agg[cr_col] / agg[tr_col], 0)
        
        if pr_col:
            agg = agg.sort_values(pr_col)
            agg['Approx Page'] = agg[pr_col].round(1)
        else:
            agg = agg.sort_values('Retention', ascending=False)
            agg['Approx Page'] = "N/A"
        agg['Milestone'] = agg[sd_col]
    else:
        df_sc = df_sc.iloc[:, :3]
        df_sc.columns = ['Milestone', 'Readers', 'Retention']
        df_sc['Retention'] = pd.to_numeric(df_sc['Retention'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df_sc['Retention'] = np.where(df_sc['Retention'] > 1, df_sc['Retention'] / 100, df_sc['Retention'])
        agg = df_sc.dropna(subset=['Retention']).copy()
        agg['Approx Page'] = "N/A"
        
    if period_name: agg['Period'] = period_name
    return agg[['Milestone', 'Retention', 'Approx Page', 'Period'] if period_name else ['Milestone', 'Retention', 'Approx Page']]


# ==============================================================================
# 🧠 DYNAMIC INSIGHT GENERATORS (WHAT, SO WHAT, NOW WHAT)
# ==============================================================================
def generate_single_insight(glo, df_prod):
    what = f"The campaign generated **{glo['views']:,.0f} views** and **{glo['clicks']:,.0f} clicks**, achieving an overall item CTR of **{(glo['clicks']/glo['views']) if glo['views']>0 else 0:.2%}**."
    
    top_cat = df_prod.groupby('Category')['Clicks'].sum().idxmax()
    top_brand = df_prod.groupby('Brand')['Clicks'].sum().idxmax()
    so_what = f"Audience engagement was heavily concentrated, with **{top_cat}** acting as the primary traffic driver for departments, and **{top_brand}** dominating brand-level affinity."
    
    now_what = f"**1.** Ensure future campaigns allocate sufficient premier page placement to {top_cat}.<br>**2.** Investigate the top 10 CTR items to identify high-performing assets that can be repurposed in future creative."
    
    return what, so_what, now_what

def generate_h2h_insight(gloA, gloB, cat_m):
    v_del = (gloB['views'] - gloA['views']) / gloA['views'] if gloA['views'] > 0 else 0
    c_del = (gloB['clicks'] - gloA['clicks']) / gloA['clicks'] if gloA['clicks'] > 0 else 0
    
    dir_v = "an increase" if v_del > 0 else "a decline"
    dir_c = "an increase" if c_del > 0 else "a decline"
    what = f"The Variant campaign saw **{dir_v} of {abs(v_del):.1%}** in total views, driving **{dir_c} of {abs(c_del):.1%}** in item clicks compared to the Base."
    
    if v_del > 0 and c_del > v_del:
        so_what = "Audience reach expanded, and engagement outpaced that growth. The assortment and pricing strategy were highly relevant to the newly acquired traffic."
    elif v_del > 0 and c_del <= 0:
        so_what = "Audience reach expanded, but overall engagement declined. This indicates traffic quality issues or an assortment that failed to resonate with the broader audience."
    elif v_del < 0 and c_del > 0:
        so_what = "Despite a smaller audience footprint, engagement actually increased. The traffic was highly qualified and the assortment was extremely relevant, but top-of-funnel reach needs addressing."
    else:
        so_what = "Both reach and engagement contracted. The flight experienced macro-level headwinds, requiring a review of both traffic acquisition and merchandising strategy."
        
    cat_m['Efficiency'] = cat_m['Alloc Variant %'] - cat_m['Alloc Base %']
    if not cat_m.empty:
        top_cat = cat_m.loc[cat_m['Allocation Shift'].idxmax()]['Category']
        now_what = f"**1. Reallocate Space:** The '{top_cat}' category saw the highest positive shift in user click share. Consider increasing its footprint in the next flyer.<br>**2. Audit Product Churn:** Review the 'YoY Assortment Turnover' table below to verify if the newly introduced SKUs actually outperformed the items retired from the Base year."
    else:
        now_what = "Review the 'YoY Assortment Turnover' to verify if the newly introduced SKUs actually outperformed the retired items."
        
    return what, so_what, now_what

def render_insight_box(what, so_what, now_what):
    st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">💡 Executive Summary & Strategic Insights</div>
            <div class="insight-text"><b>What Happened:</b> {what}</div>
            <div class="insight-text"><b>So What:</b> {so_what}</div>
            <div class="insight-text"><b>Now What:</b> {now_what}</div>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 🗂️ MODULE 1: SINGLE CAMPAIGN MATRIX
# ==============================================================================
def render_single_campaign_matrix():
    st.markdown("<div class='main-header'>Single Campaign Strategy Matrix</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload raw exports directly to map campaign performance.</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1: merch_file = st.file_uploader("Upload Merchandise Performance File (.xlsx/.csv)", type=["xlsx", "csv"])
    with col2: scroll_file = st.file_uploader("Upload Scroll Depth File [Optional] (.xlsx/.csv)", type=["xlsx", "csv"])
        
    if merch_file:
        df_clean, m, header_idx = scrub_and_load_excel(merch_file)
        
        if df_clean is not None:
            df_prod, df_creative, global_totals = process_metrics(df_clean, m)
            merchant, run_name, run_id, date_from, date_to = extract_exact_metadata(df_clean)
            
            st.info(f"📍 **ACTIVE FLIGHT RECAP:** {merchant}  |  **Flight Group:** {run_name} (ID: {run_id})  |  **Window:** {date_from} to {date_to}")
            
            # 🧠 RENDER DYNAMIC INSIGHTS
            w, sw, nw = generate_single_insight(global_totals, df_prod)
            render_insight_box(w, sw, nw)
            
            v_tot, cl_tot, cp_tot, t_tot = global_totals['views'], global_totals['clicks'], global_totals['clips'], global_totals['ttms']
            ctr_global_display = f"{cl_tot/v_tot:.2%}" if v_tot > 0 else "0.00%"
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(f"<div class='metric-card'><div class='metric-val'>{v_tot:,.0f}</div><div class='metric-lbl'>TOTAL VIEWS</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><div class='metric-val'>{cl_tot:,.0f}</div><div class='metric-lbl'>TOTAL CLICKS</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><div class='metric-val'>{cp_tot:,.0f}</div><div class='metric-lbl'>ADD TO LISTS</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><div class='metric-val'>{t_tot:,.0f}</div><div class='metric-lbl'>TOTAL TTMS</div></div>", unsafe_allow_html=True)
            c5.markdown(f"<div class='metric-card'><div class='metric-val'>{ctr_global_display}</div><div class='metric-lbl'>TOTAL ITEM CTR</div></div>", unsafe_allow_html=True)
            
            st.write("---")
            st.subheader("🏆 Top 10 Items by Performance CTR")
            pivot_top = df_prod.groupby('SKU').agg({'Name': 'first', 'Page': 'first', 'Views': 'sum', 'Clicks': 'sum', 'Clips': 'sum', 'TTMs': 'sum'}).reset_index()
            pivot_top['Item CTR'] = np.where(pivot_top['Views'] > 0, pivot_top['Clicks'] / pivot_top['Views'], 0.0)
            st.dataframe(pivot_top[['SKU', 'Name', 'Page', 'Views', 'Clicks', 'Clips', 'TTMs', 'Item CTR']].sort_values(by='Item CTR', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)
            
            st.write("---")
            st.subheader("📊 Item Allocation vs Click Share")
            cat_agg = df_prod.groupby('Category').agg(Count=('SKU', 'count'), Views=('Views', 'sum'), Clicks=('Clicks', 'sum'), Clips=('Clips', 'sum'), TTMs=('TTMs', 'sum')).reset_index()
            cat_agg['Item Allocation %'] = cat_agg['Count'] / cat_agg['Count'].sum() if cat_agg['Count'].sum() > 0 else 0
            cat_agg['Click Share %'] = cat_agg['Clicks'] / cat_agg['Clicks'].sum() if cat_agg['Clicks'].sum() > 0 else 0
            
            col_table, col_chart = st.columns(2)
            with col_table: st.dataframe(cat_agg.sort_values(by='Clicks', ascending=False).style.format({'Count': '{:,.0f}', 'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item Allocation %': '{:.1%}', 'Click Share %': '{:.1%}'}), use_container_width=True, hide_index=True)
            with col_chart: st.plotly_chart(px.bar(cat_agg.melt(id_vars='Category', value_vars=['Item Allocation %', 'Click Share %']), x='Category', y='value', color='variable', barmode='group', color_discrete_sequence=['#0054B7', '#43c4f4'], title="Category Share Allocation"), use_container_width=True)

            st.write("---")
            st.subheader("🏬 Holistic Brand Affinity & Marketing Summary")
            b_col, c_col = st.columns(2)
            with b_col:
                st.markdown("**Holistic Brand Performance Matrix**")
                brand_agg = df_prod.groupby('Brand').agg(Unique_Items=('SKU', 'nunique'), Views=('Views','sum'), Clicks=('Clicks','sum'), TTMs=('TTMs','sum')).reset_index()
                brand_agg['Brand Transfer %'] = brand_agg['TTMs'] / t_tot if t_tot > 0 else 0
                st.dataframe(brand_agg.sort_values(by='Clicks', ascending=False).head(15).style.format({'Unique_Items': '{:,.0f}', 'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'TTMs': '{:,.0f}', 'Brand Transfer %': '{:.1%}'}), use_container_width=True, hide_index=True)
            with c_col:
                st.markdown("**Creative Marketing Asset Summary (Display Type: LINK)**")
                if not df_creative.empty:
                    cr_agg = df_creative.groupby('Name').agg(Page=('Page','max'), Views=('Views','sum'), Clicks=('Clicks','sum')).reset_index()
                    cr_agg['Asset CTR'] = np.where(cr_agg['Views'] > 0, cr_agg['Clicks'] / cr_agg['Views'], 0)
                    st.dataframe(cr_agg.sort_values(by='Clicks', ascending=False).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Asset CTR': '{:.2%}'}), use_container_width=True, hide_index=True)

            if scroll_file:
                st.write("---")
                st.subheader("📉 Audience Scroll Retention & Drop-off")
                try:
                    df_sc = process_scroll_file(scroll_file)
                    sc_col1, sc_col2 = st.columns([1, 2])
                    with sc_col1:
                        st.markdown("**Retention Drop-off Table**")
                        df_sc_table = df_sc.copy().rename(columns={'Milestone': 'Scroll Depth', 'Retention': '% of Users Read'})
                        st.dataframe(df_sc_table[['Scroll Depth', '% of Users Read', 'Approx Page']].style.format({'% of Users Read': '{:.1%}'}), use_container_width=True, hide_index=True)
                    with sc_col2:
                        st.markdown("**Retention Drop-off Curve**")
                        fig = px.line(df_sc, x='Milestone', y='Retention', markers=True, color_discrete_sequence=['#0054B7'], labels={'Milestone': 'Scroll Depth', 'Retention': '% of Users Read'})
                        fig.update_layout(yaxis=dict(tickformat='.0%', range=[0,1]))
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not process the scroll file. Error: {str(e)}")

# ==============================================================================
# 🗂️ MODULE 2: HEAD-TO-HEAD VARIANCE
# ==============================================================================
def render_head_to_head_variance():
    st.markdown("<div class='main-header'>Head-to-Head YoY Variance Report Matrix</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Compare Period A (Base Year) against Period B (Variant Year) to calculate strategic growth deltas.</div>", unsafe_allow_html=True)
    
    colA, colB = st.columns(2)
    with colA: 
        file_A = st.file_uploader("📁 Period A: Base File (Control Year)", type=["xlsx", "csv"])
        scroll_A = st.file_uploader("📉 Period A: Scroll Depth File [Optional]", type=["xlsx", "csv"])
    with colB: 
        file_B = st.file_uploader("📁 Period B: Variant File (Test Year)", type=["xlsx", "csv"])
        scroll_B = st.file_uploader("📉 Period B: Scroll Depth File [Optional]", type=["xlsx", "csv"])
    
    if not file_A or not file_B:
        st.warning("⚠️ **Waiting for data:** Please upload BOTH the Base File and the Variant File to generate the comparison matrix.")
        return
        
    dfA_clean, mA, _ = scrub_and_load_excel(file_A)
    dfB_clean, mB, _ = scrub_and_load_excel(file_B)
    
    if dfA_clean is not None and dfB_clean is not None:
        dfA_prod, _, gloA = process_metrics(dfA_clean, mA)
        dfB_prod, _, gloB = process_metrics(dfB_clean, mB)
        
        _, rA, _, dA_from, dA_to = extract_exact_metadata(dfA_clean)
        _, rB, _, dB_from, dB_to = extract_exact_metadata(dfB_clean)
        
        st.info(f"⚖️ **COMPARING:** {rA} ({dA_from} to {dA_to}) **VERSUS** {rB} ({dB_from} to {dB_to})")
        
        catA = dfA_prod.groupby('Category').agg(CntA=('SKU', 'count'), ClkA=('Clicks', 'sum')).reset_index()
        catA['Alloc Base %'] = catA['CntA'] / catA['CntA'].sum() if catA['CntA'].sum() > 0 else 0
        catB = dfB_prod.groupby('Category').agg(CntB=('SKU', 'count'), ClkB=('Clicks', 'sum')).reset_index()
        catB['Alloc Variant %'] = catB['CntB'] / catB['CntB'].sum() if catB['CntB'].sum() > 0 else 0
        cat_m = pd.merge(catA[['Category', 'Alloc Base %']], catB[['Category', 'Alloc Variant %']], on='Category', how='outer').fillna(0)
        cat_m['Allocation Shift'] = cat_m['Alloc Variant %'] - cat_m['Alloc Base %']

        # 🧠 RENDER DYNAMIC INSIGHTS
        w, sw, nw = generate_h2h_insight(gloA, gloB, cat_m)
        render_insight_box(w, sw, nw)
        
        def calc_delta(base, var): return 1.0 if base == 0 and var > 0 else (0.0 if base == 0 else (var - base) / base)

        st.write("---")
        st.subheader("🎯 Slot 1: Macro Funnel Delta")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("TOTAL VIEWS", f"{gloB['views']:,.0f}", f"{calc_delta(gloA['views'], gloB['views']):.1%}")
        c2.metric("TOTAL CLICKS", f"{gloB['clicks']:,.0f}", f"{calc_delta(gloA['clicks'], gloB['clicks']):.1%}")
        c3.metric("ADD TO LISTS", f"{gloB['clips']:,.0f}", f"{calc_delta(gloA['clips'], gloB['clips']):.1%}")
        c4.metric("TOTAL TTMS", f"{gloB['ttms']:,.0f}", f"{calc_delta(gloA['ttms'], gloB['ttms']):.1%}")
        ctrA, ctrB = (gloA['clicks']/gloA['views'] if gloA['views'] > 0 else 0), (gloB['clicks']/gloB['views'] if gloB['views'] > 0 else 0)
        c5.metric("GLOBAL ITEM CTR", f"{ctrB:.2%}", f"{ctrB - ctrA:+.2%} pts")

        st.write("---")
        st.subheader("🏬 Slot 2: Brand Momentum Winners & Losers")
        brA, brB = dfA_prod.groupby('Brand')[['Views', 'Clicks', 'TTMs']].sum().reset_index(), dfB_prod.groupby('Brand')[['Views', 'Clicks', 'TTMs']].sum().reset_index()
        br_merge = pd.merge(brA, brB, on='Brand', suffixes=(' Base', ' Variant'), how='outer').fillna(0)
        br_merge['TTM Growth %'] = np.where(br_merge['TTMs Base'] > 0, (br_merge['TTMs Variant'] - br_merge['TTMs Base']) / br_merge['TTMs Base'], np.where(br_merge['TTMs Variant'] > 0, 1.0, 0.0))
        br_merge['Click Growth %'] = np.where(br_merge['Clicks Base'] > 0, (br_merge['Clicks Variant'] - br_merge['Clicks Base']) / br_merge['Clicks Base'], np.where(br_merge['Clicks Variant'] > 0, 1.0, 0.0))
        st.dataframe(br_merge[['Brand', 'TTMs Base', 'TTMs Variant', 'TTM Growth %', 'Clicks Base', 'Clicks Variant', 'Click Growth %']].sort_values(by='TTM Growth %', ascending=False).style.format({'TTMs Base': '{:,.0f}', 'TTMs Variant': '{:,.0f}', 'Clicks Base': '{:,.0f}', 'Clicks Variant': '{:,.0f}', 'TTM Growth %': '{:+.1%}', 'Click Growth %': '{:+.1%}'}), use_container_width=True, hide_index=True)

        st.write("---")
        st.subheader("📊 Slot 3: Category Share Shifts")
        st.dataframe(cat_m.sort_values(by='Allocation Shift', ascending=False).style.format({'Alloc Base %': '{:.1%}', 'Alloc Variant %': '{:.1%}', 'Allocation Shift': '{:+.2%} pts'}), use_container_width=True, hide_index=True)

        st.write("---")
        st.subheader("🏆 Slot 4: Shared SKU Micro-Delta")
        sk_m = pd.merge(dfA_prod.groupby('SKU').agg({'Name': 'first', 'Views': 'sum', 'Clicks': 'sum', 'Curr_Price': 'mean'}).reset_index(), dfB_prod.groupby('SKU').agg({'Name': 'first', 'Views': 'sum', 'Clicks': 'sum', 'Curr_Price': 'mean'}).reset_index(), on='SKU', suffixes=(' Base', ' Variant'), how='inner')
        if not sk_m.empty:
            sk_m['CTR Base'] = np.where(sk_m['Views Base'] > 0, sk_m['Clicks Base'] / sk_m['Views Base'], 0)
            sk_m['CTR Variant'] = np.where(sk_m['Views Variant'] > 0, sk_m['Clicks Variant'] / sk_m['Views Variant'], 0)
            sk_m['CTR Shift'], sk_m['Price Shift'] = sk_m['CTR Variant'] - sk_m['CTR Base'], sk_m['Curr_Price Variant'] - sk_m['Curr_Price Base']
            st.dataframe(sk_m[['SKU', 'Name Variant', 'Views Variant', 'Clicks Variant', 'CTR Base', 'CTR Variant', 'CTR Shift', 'Price Shift']].rename(columns={'Name Variant': 'Name'}).sort_values(by='CTR Shift', ascending=False).style.format({'Views Variant': '{:,.0f}', 'Clicks Variant': '{:,.0f}', 'CTR Base': '{:.2%}', 'CTR Variant': '{:.2%}', 'CTR Shift': '{:+.2%} pts', 'Price Shift': '${:+.2f}'}), use_container_width=True, hide_index=True)
            
        st.write("---")
        st.subheader("🔄 Slot 5: YoY Assortment Turnover")
        col_new, col_ret = st.columns(2)
        with col_new:
            st.markdown("**Top New Items**")
            new_skus = dfB_prod[~dfB_prod['SKU'].isin(dfA_prod['SKU'])].groupby('SKU').agg({'Name': 'first', 'Views': 'sum', 'Clicks': 'sum', 'TTMs': 'sum'}).reset_index()
            new_skus['Item CTR'] = np.where(new_skus['Views'] > 0, new_skus['Clicks'] / new_skus['Views'], 0)
            st.dataframe(new_skus.sort_values(by='Clicks', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True) if not new_skus.empty else st.caption("No new items.")
        with col_ret:
            st.markdown("**Top Retired Items**")
            ret_skus = dfA_prod[~dfA_prod['SKU'].isin(dfB_prod['SKU'])].groupby('SKU').agg({'Name': 'first', 'Views': 'sum', 'Clicks': 'sum', 'TTMs': 'sum'}).reset_index()
            ret_skus['Item CTR'] = np.where(ret_skus['Views'] > 0, ret_skus['Clicks'] / ret_skus['Views'], 0)
            st.dataframe(ret_skus.sort_values(by='Clicks', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True) if not ret_skus.empty else st.caption("No items retired.")

        st.write("---")
        st.subheader("💰 Slot 6: YoY Pricing & Promotional Shift")
        for d in [dfA_prod, dfB_prod]: 
            d['Price_Tier'] = pd.cut(d['Curr_Price'], bins=[0, 25, 50, 100, 250, 500, float('inf')], labels=["Under $25", "$25 - $50", "$50 - $100", "$100 - $250", "$250 - $500", "$500+"])
            d['Discount_Tier'] = pd.cut(d['Discount_Pct'], bins=[-1, 0, 15, 30, 50, float('inf')], labels=["No Discount", "1% - 15%", "16% - 30%", "31% - 50%", "50%+"])
        
        pA, pB = dfA_prod.groupby('Price_Tier', observed=False)['Clicks'].sum().reset_index().rename(columns={'Clicks': 'Base Clicks'}), dfB_prod.groupby('Price_Tier', observed=False)['Clicks'].sum().reset_index().rename(columns={'Clicks': 'Variant Clicks'})
        p_merge = pd.merge(pA, pB, on='Price_Tier').fillna(0)
        p_merge['Click Share Shift'] = (p_merge['Variant Clicks'] / p_merge['Variant Clicks'].sum()) - (p_merge['Base Clicks'] / p_merge['Base Clicks'].sum())
        
        dA, dB = dfA_prod.groupby('Discount_Tier', observed=False)['Clicks'].sum().reset_index().rename(columns={'Clicks': 'Base Clicks'}), dfB_prod.groupby('Discount_Tier', observed=False)['Clicks'].sum().reset_index().rename(columns={'Clicks': 'Variant Clicks'})
        d_merge = pd.merge(dA, dB, on='Discount_Tier').fillna(0)
        d_merge['Click Share Shift'] = (d_merge['Variant Clicks'] / d_merge['Variant Clicks'].sum()) - (d_merge['Base Clicks'] / d_merge['Base Clicks'].sum())
        
        c_p, c_d = st.columns(2)
        with c_p: st.dataframe(p_merge.style.format({'Base Clicks': '{:,.0f}', 'Variant Clicks': '{:,.0f}', 'Click Share Shift': '{:+.2%}'}), use_container_width=True, hide_index=True)
        with c_d: st.dataframe(d_merge.style.format({'Base Clicks': '{:,.0f}', 'Variant Clicks': '{:,.0f}', 'Click Share Shift': '{:+.2%}'}), use_container_width=True, hide_index=True)

        if scroll_A and scroll_B:
            st.write("---")
            st.subheader("📉 Slot 7: YoY Audience Scroll Retention")
            try:
                df_scA, df_scB = process_scroll_file(scroll_A, 'Base Year (Period A)'), process_scroll_file(scroll_B, 'Variant Year (Period B)')
                sc_col1, sc_col2 = st.columns([1, 2])
                with sc_col1:
                    tbl_merge = pd.merge(df_scA[['Milestone', 'Approx Page', 'Retention']].rename(columns={'Retention': 'Base % Read', 'Approx Page': 'Base Page'}), df_scB[['Milestone', 'Approx Page', 'Retention']].rename(columns={'Retention': 'Variant % Read', 'Approx Page': 'Variant Page'}), on='Milestone', how='outer').rename(columns={'Milestone': 'Scroll Depth'})
                    tbl_merge['Approx Page'] = tbl_merge['Variant Page'].combine_first(tbl_merge['Base Page'])
                    st.dataframe(tbl_merge[['Scroll Depth', 'Base % Read', 'Variant % Read', 'Approx Page']].style.format({'Base % Read': '{:.1%}', 'Variant % Read': '{:.1%}'}), use_container_width=True, hide_index=True)
                with sc_col2:
                    st.plotly_chart(px.line(pd.concat([df_scA, df_scB]), x='Milestone', y='Retention', color='Period', markers=True, color_discrete_sequence=['#475569', '#0054B7'], labels={'Milestone': 'Scroll Depth', 'Retention': '% of Users Read'}).update_layout(yaxis=dict(tickformat='.0%', range=[0,1])), use_container_width=True)
            except Exception as e: st.warning(f"Could not process scroll files. Error: {str(e)}")

# ==============================================================================
# 🗺️ NAVIGATION & MAIN APP CONTROL
# ==============================================================================
st.sidebar.markdown("<h2 style='color:#002551;'>🚀 Control Panel</h2>", unsafe_allow_html=True)
pipeline_mode = st.sidebar.radio("Select Strategy Module:", ["📁 Single Campaign Matrix", "📊 Head-to-Head Variance"])
if pipeline_mode == "📁 Single Campaign Matrix": render_single_campaign_matrix()
elif pipeline_mode == "📊 Head-to-Head Variance": render_head_to_head_variance()