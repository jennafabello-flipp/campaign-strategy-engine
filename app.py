import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import re
import os

# Create the hidden benchmarks directory on the server if it doesn't exist
if not os.path.exists("benchmarks"):
    os.makedirs("benchmarks")

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

def scrub_and_load_excel(file_obj, is_local_path=False):
    if file_obj is None: return None, None, None
    try:
        if is_local_path:
            is_csv = file_obj.lower().endswith('.csv')
            with open(file_obj, 'rb') as f:
                file_bytes = f.read()
        else:
            file_bytes = file_obj.read()
            is_csv = file_obj.name.lower().endswith('.csv')
            
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
            'date': get_col(['Daily Available From', 'Date', 'Start Date']),
            'run_id': get_col(['Flyer Run ID', 'Run ID', 'Campaign ID']), # NEW RUN ID CATCHER
            'display_type': get_col(['Display Type']), 'page': get_col(['Page Position', 'Page']),
            'brand': get_col(['Brand', 'Manufacturer']), 'orig_price': get_col(['Total Original Price', 'Original Price']),
            'curr_price': get_col(['Total Current Price', 'Current Price']), 'url': get_col(['URL', 'Destination URL', 'Link', 'Destination Link']),
            'c1': get_col(['Custom ID 1']), 'c2': get_col(['Custom ID 2']), 'c3': get_col(['Custom ID 3']),
            'ret_cat': get_col(['Retailer Category']), 'goo_l1': get_col(['Google Category L1']), 'goo_l2': get_col(['Google Category L2']), 'goo_l3': get_col(['Google Category L3']),
            'views': get_col(['Total Item Views', 'Views']), 'clicks': get_col(['Total Item Clicks', 'Clicks']),
            'clips': get_col(['Total Clippings', 'Clips']), 'ttms': get_col(['Total Transfer to Merchant (TTMs)', 'Total Transfer to Merchant', 'TTMS'])
        }
        return df_clean, mapping, header_idx
    except Exception as e:
        st.error(f"Error scrubbing file setup: {str(e)}")
        return None, None, None

def process_metrics(df, m):
    df['Name'] = df[m['name']].astype(str).str.strip().apply(clean_bilingual_suffix) if m['name'] else "Unnamed Asset"
    df['Display_Type'] = df[m['display_type']].astype(str).str.upper().str.strip() if m['display_type'] else "PRODUCT"
    df['Page'] = df[m['page']].astype(str).str.extract(r'(\d+)').fillna(1).astype(int) if m['page'] else 1
    df['Brand'] = df[m['brand']].astype(str).str.strip() if m['brand'] and m['brand'] in df.columns else "UNKNOWN"
    df['Date'] = pd.to_datetime(df[m['date']], errors='coerce') if m.get('date') else pd.NaT
    df['Run_ID'] = df[m['run_id']].astype(str) if m.get('run_id') else "UNKNOWN" # STORING RUN ID FOR PAGE AVERAGES
    
    def safe_numeric(col_name):
        if m[col_name] and m[col_name] in df.columns:
            cleaned = df[m[col_name]].astype(str).str.replace(r'[^\d.]', '', regex=True).replace('', '0')
            return pd.to_numeric(cleaned, errors='coerce').fillna(0)
        return 0

    df['Views'], df['Clicks'], df['Clips'], df['TTMs'] = safe_numeric('views'), safe_numeric('clicks'), safe_numeric('clips'), safe_numeric('ttms')
    df['Orig_Price'], df['Curr_Price'] = safe_numeric('orig_price'), safe_numeric('curr_price')
    df['Discount_Pct'] = np.where(df['Orig_Price'] > 0, ((df['Orig_Price'] - df['Curr_Price']) / df['Orig_Price']) * 100, 0.0)
    df['Discount_Pct'] = np.where(df['Discount_Pct'] < 0, 0.0, df['Discount_Pct'])

    is_sku_clone = df['Brand'].isin(['nan', 'NaN', 'None', '', 'UNKNOWN'])
    if m['sku'] and m['sku'] in df.columns:
        is_sku_clone = is_sku_clone | (df['Brand'] == df[m['sku']])
        
    df.loc[is_sku_clone, 'Brand'] = df.loc[is_sku_clone, 'Name'].apply(lambda x: str(x).split()[0].upper() if str(x).strip() != "" else "GENERIC")

    def normalize_sku(row):
        s = str(row[m['sku']]).strip() if m['sku'] else "UNKNOWN"
        if s.endswith('.0'): s = s[:-2]
        if s.lower() not in ['nan', 'none', '', 'null', '0', 'unknown']: return s
        if m.get('url') and pd.notna(row[m['url']]):
            url = str(row[m['url']])
            match = re.search(r'(?:variantCode|sku|id|pid)=([A-Za-z0-9_-]+)', url, re.IGNORECASE)
            if match: return f"URL_{match.group(1).upper()}"
            match_p = re.search(r'/p/([A-Za-z0-9_-]+)', url, re.IGNORECASE)
            if match_p: return f"URL_{match_p.group(1).upper()}"
            
        brand_clean = str(row['Brand']).strip().upper()
        page_clean = str(row['Page'])
        price_clean = str(row['Curr_Price'])
        fingerprint = f"{brand_clean}_PG{page_clean}_{price_clean}"
        if fingerprint != "GENERIC_PG1_0.0" and fingerprint != "UNKNOWN_PG1_0.0":
            return fingerprint
        return str(row['Name']).upper()
        
    df['SKU'] = df.apply(normalize_sku, axis=1)

    def get_l1(row):
        for key in ['c1', 'ret_cat', 'goo_l1']:
            if m[key] and pd.notna(row[m[key]]):
                val = str(row[m[key]]).strip()
                if val not in ["", "NULL", "nan", "NaN", "None"]: return val
        return "General Merchandise"

    def get_l2(row):
        for key in ['c2', 'goo_l2']:
            if m[key] and pd.notna(row[m[key]]):
                val = str(row[m[key]]).strip()
                if val not in ["", "NULL", "nan", "NaN", "None"]: return val
        return "Uncategorized Sub-Department"
        
    def get_l3(row):
        for key in ['c3', 'goo_l3']:
            if m[key] and pd.notna(row[m[key]]):
                val = str(row[m[key]]).strip()
                if val not in ["", "NULL", "nan", "NaN", "None"]: return val
        return "Uncategorized Item-Level"

    df['L1_Category'] = df.apply(get_l1, axis=1)
    df['L2_Category'] = df.apply(get_l2, axis=1)
    df['L3_Category'] = df.apply(get_l3, axis=1)
    
    global_totals = {'views': df['Views'].sum(), 'clicks': df['Clicks'].sum(), 'clips': df['Clips'].sum(), 'ttms': df['TTMs'].sum()}
    
    is_marketing_link = (df['Display_Type'] == "LINK") | (df['Name'].str.contains('BANNER', case=False, na=False))
    return df[~is_marketing_link].copy(), df[is_marketing_link].copy(), global_totals

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
            
            top_cat = df_prod.groupby('L1_Category')['Clicks'].sum().idxmax() if not df_prod.empty else "General Merchandise"
            top_brand = df_prod.groupby('Brand')['Clicks'].sum().idxmax() if not df_prod.empty else "UNKNOWN"
            render_insight_box(
                f"The campaign generated **{global_totals['views']:,.0f} views** and **{global_totals['clicks']:,.0f} clicks**, achieving an overall item CTR of **{(global_totals['clicks']/global_totals['views']) if global_totals['views']>0 else 0:.2%}**.",
                f"Audience engagement was heavily concentrated, with **{top_cat}** acting as the primary traffic driver for departments, and **{top_brand}** dominating brand-level affinity.",
                f"**1.** Ensure future campaigns allocate sufficient premier page placement to {top_cat}.<br>**2.** Investigate the top 10 CTR items to identify high-performing assets that can be repurposed in future creative."
            )
            
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
            def build_cat_agg(cat_col):
                c_agg = df_prod.groupby(cat_col).agg(Count=('SKU', 'count'), Views=('Views', 'sum'), Clicks=('Clicks', 'sum'), Clips=('Clips', 'sum'), TTMs=('TTMs', 'sum')).reset_index()
                c_agg['Item Allocation %'] = c_agg['Count'] / c_agg['Count'].sum() if c_agg['Count'].sum() > 0 else 0
                c_agg['Click Share %'] = c_agg['Clicks'] / c_agg['Clicks'].sum() if c_agg['Clicks'].sum() > 0 else 0
                return c_agg

            cat_l1_agg, cat_l2_agg, cat_l3_agg = build_cat_agg('L1_Category'), build_cat_agg('L2_Category'), build_cat_agg('L3_Category')
            tab_l1, tab_l2, tab_l3 = st.tabs(["L1 Primary Category", "L2 Subcategory", "L3 Sub-subcategory"])
            
            with tab_l1:
                col_t1, col_c1 = st.columns(2)
                with col_t1: st.dataframe(cat_l1_agg.sort_values(by='Clicks', ascending=False).style.format({'Count': '{:,.0f}', 'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item Allocation %': '{:.1%}', 'Click Share %': '{:.1%}'}), use_container_width=True, hide_index=True)
                with col_c1: st.plotly_chart(px.bar(cat_l1_agg.melt(id_vars='L1_Category', value_vars=['Item Allocation %', 'Click Share %']), x='L1_Category', y='value', color='variable', barmode='group', color_discrete_sequence=['#0054B7', '#43c4f4'], title="L1 Category Share Allocation"), use_container_width=True)
            with tab_l2:
                col_t2, col_c2 = st.columns(2)
                with col_t2: st.dataframe(cat_l2_agg.sort_values(by='Clicks', ascending=False).style.format({'Count': '{:,.0f}', 'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item Allocation %': '{:.1%}', 'Click Share %': '{:.1%}'}), use_container_width=True, hide_index=True)
                with col_c2: st.plotly_chart(px.bar(cat_l2_agg.melt(id_vars='L2_Category', value_vars=['Item Allocation %', 'Click Share %']), x='L2_Category', y='value', color='variable', barmode='group', color_discrete_sequence=['#0054B7', '#43c4f4'], title="L2 Subcategory Share Allocation"), use_container_width=True)
            with tab_l3:
                col_t3, col_c3 = st.columns(2)
                with col_t3: st.dataframe(cat_l3_agg.sort_values(by='Clicks', ascending=False).style.format({'Count': '{:,.0f}', 'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item Allocation %': '{:.1%}', 'Click Share %': '{:.1%}'}), use_container_width=True, hide_index=True)
                with col_c3: st.plotly_chart(px.bar(cat_l3_agg.melt(id_vars='L3_Category', value_vars=['Item Allocation %', 'Click Share %']), x='L3_Category', y='value', color='variable', barmode='group', color_discrete_sequence=['#0054B7', '#43c4f4'], title="L3 Sub-subcategory Share Allocation"), use_container_width=True)

            st.write("---")
            st.subheader("🏬 Holistic Brand Affinity & Marketing Summary")
            b_col, c_col = st.columns(2)
            with b_col:
                brand_agg = df_prod.groupby('Brand').agg(Unique_Items=('SKU', 'nunique'), Views=('Views','sum'), Clicks=('Clicks','sum'), Clips=('Clips','sum'), TTMs=('TTMs','sum')).reset_index()
                brand_agg['Click Share %'] = brand_agg['Clicks'] / cl_tot if cl_tot > 0 else 0
                brand_agg['List Share %'] = brand_agg['Clips'] / cp_tot if cp_tot > 0 else 0
                brand_agg['TTM Share %'] = brand_agg['TTMs'] / t_tot if t_tot > 0 else 0
                st.dataframe(brand_agg[['Brand', 'Unique_Items', 'Clicks', 'Click Share %', 'Clips', 'List Share %', 'TTMs', 'TTM Share %']].sort_values(by='Clicks', ascending=False).head(15).style.format({
                    'Unique_Items': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 
                    'Click Share %': '{:.2%}', 'List Share %': '{:.2%}', 'TTM Share %': '{:.2%}'
                }), use_container_width=True, hide_index=True)
            with c_col:
                if not df_creative.empty:
                    cr_agg = df_creative.groupby('Name').agg(Page=('Page','max'), Views=('Views','sum'), Clicks=('Clicks','sum')).reset_index()
                    cr_agg['Asset CTR'] = np.where(cr_agg['Views'] > 0, cr_agg['Clicks'] / cr_agg['Views'], 0)
                    st.dataframe(cr_agg.sort_values(by='Clicks', ascending=False).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Asset CTR': '{:.2%}'}), use_container_width=True, hide_index=True)

            st.write("---")
            st.subheader("💰 Pricing & Promotional Band Analysis")
            df_prod_bands = df_prod.copy()
            df_prod_bands['Price_Tier'] = pd.cut(df_prod_bands['Curr_Price'], bins=[-1, 25, 50, 100, 250, 500, float('inf')], labels=["Under $25", "$25 - $50", "$50 - $100", "$100 - $250", "$250 - $500", "$500+"])
            df_prod_bands['Discount_Tier'] = pd.cut(df_prod_bands['Discount_Pct'], bins=[-1, 0, 15, 30, 50, float('inf')], labels=["No Discount", "1% - 15%", "16% - 30%", "31% - 50%", "50%+"])
            
            p_agg = df_prod_bands.groupby('Price_Tier', observed=False).agg(Items=('SKU', 'nunique'), Clicks=('Clicks', 'sum'), Clips=('Clips', 'sum'), TTMs=('TTMs', 'sum')).reset_index()
            p_agg['Click Share %'] = p_agg['Clicks'] / cl_tot if cl_tot > 0 else 0
            p_agg['List Share %'] = p_agg['Clips'] / cp_tot if cp_tot > 0 else 0
            p_agg['TTM Share %'] = p_agg['TTMs'] / t_tot if t_tot > 0 else 0
            
            d_agg = df_prod_bands.groupby('Discount_Tier', observed=False).agg(Items=('SKU', 'nunique'), Clicks=('Clicks', 'sum'), Clips=('Clips', 'sum'), TTMs=('TTMs', 'sum')).reset_index()
            d_agg['Click Share %'] = d_agg['Clicks'] / cl_tot if cl_tot > 0 else 0
            d_agg['List Share %'] = d_agg['Clips'] / cp_tot if cp_tot > 0 else 0
            d_agg['TTM Share %'] = d_agg['TTMs'] / t_tot if t_tot > 0 else 0
            
            band_fmt = {'Items': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 'Click Share %': '{:.2%}', 'List Share %': '{:.2%}', 'TTM Share %': '{:.2%}'}
            c_p, c_d = st.columns(2)
            with c_p: st.dataframe(p_agg[['Price_Tier', 'Items', 'Clicks', 'Click Share %', 'Clips', 'List Share %', 'TTMs', 'TTM Share %']].sort_values(by='Clicks', ascending=False).style.format(band_fmt), use_container_width=True, hide_index=True)
            with c_d: st.dataframe(d_agg[['Discount_Tier', 'Items', 'Clicks', 'Click Share %', 'Clips', 'List Share %', 'TTMs', 'TTM Share %']].sort_values(by='Clicks', ascending=False).style.format(band_fmt), use_container_width=True, hide_index=True)

            if scroll_file:
                st.write("---")
                st.subheader("📉 Audience Scroll Retention & Drop-off")
                try:
                    df_sc = process_scroll_file(scroll_file)
                    sc_col1, sc_col2 = st.columns([1, 2])
                    with sc_col1:
                        df_sc_table = df_sc.copy().rename(columns={'Milestone': 'Scroll Depth', 'Retention': '% of Users Read'})
                        st.dataframe(df_sc_table[['Scroll Depth', '% of Users Read', 'Approx Page']].style.format({'% of Users Read': '{:.1%}'}), use_container_width=True, hide_index=True)
                    with sc_col2:
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
    with colA: file_A = st.file_uploader("📁 Period A: Base File (Control Year)", type=["xlsx", "csv"])
    with colB: file_B = st.file_uploader("📁 Period B: Variant File (Test Year)", type=["xlsx", "csv"])
    
    if not file_A or not file_B:
        st.warning("⚠️ **Waiting for data:** Please upload BOTH the Base File and the Variant File to generate the comparison matrix.")
        return
        
    dfA_clean, mA, _ = scrub_and_load_excel(file_A)
    dfB_clean, mB, _ = scrub_and_load_excel(file_B)
    
    if dfA_clean is not None and dfB_clean is not None:
        dfA_prod, _, gloA = process_metrics(dfA_clean, mA)
        dfB_prod, _, gloB = process_metrics(dfB_clean, mB)
        
        st.success("✅ Head-to-Head successfully processed. (Run Single Campaign or Benchmark scorecard to see expanded analytics!)")

# ==============================================================================
# 🏆 MODULE 3: YEARLY BENCHMARK SCORECARD
# ==============================================================================
def render_benchmark_scorecard():
    st.markdown("<div class='main-header'>🏆 Yearly Benchmark Scorecard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Compare a client's current flight directly against a historical industry baseline, aligned by season.</div>", unsafe_allow_html=True)
    
    colA, colB = st.columns(2)
    with colA:
        client_file = st.file_uploader("📁 Upload Client File (Current Campaign)", type=["xlsx", "csv"])
    with colB:
        # 🧠 THE NEW DICTIONARY: Pretty Dropdown Name -> Exact File Name
        benchmark_map = {
            "🛒 Grocery (Core: Jan-Oct)": "Grocery_Core",
            "🎄 Grocery (Holiday: Nov-Dec)": "Grocery_Holiday",
            "🛒 Pharmacy (Core: Jan-Oct)": "Pharmacy_Core",
            "🎄 Pharmacy (Holiday: Nov-Dec)": "Pharmacy_Holiday",
            "💻 Electronics (Core: Jan-Oct)": "Electronics_Core",
            "🎄 Electronics (Holiday: Nov-Dec)": "Electronics_Holiday",
            "🛠️ Home Improvement (Core: Jan-Oct)": "Home_Improvement_Core",
            "🎄 Home Improvement (Holiday: Nov-Dec)": "Home_Improvement_Holiday",
            "🐾 Pet Care (Core: Jan-Oct)": "Pet_Care_Core",
            "🎄 Pet Care (Holiday: Nov-Dec)": "Pet_Care_Holiday",
            "🛋️ Home Goods & Furniture (Core: Jan-Oct)": "Home_Goods_Core",
            "🎄 Home Goods & Furniture (Holiday: Nov-Dec)": "Home_Goods_Holiday",
            "📦 General Merchandise (Core: Jan-Oct)": "General_Merchandise_Core",
            "🎄 General Merchandise (Holiday: Nov-Dec)": "General_Merchandise_Holiday",
            "📎 Office Supplies (Core: Jan-Oct)": "Office_Supplies_Core",
            "🎄 Office Supplies (Holiday: Nov-Dec)": "Office_Supplies_Holiday"
        }
        
        selected_option = st.selectbox("🎯 Select Industry Baseline", list(benchmark_map.keys()))
        
    if not client_file:
        st.warning("⚠️ **Waiting for data:** Please upload the Client File to generate the Benchmark Scorecard.")
        return
        
    # 1. Search for the exact file name mapped in the dictionary
    exact_file_name = benchmark_map[selected_option]
    base_path = f"benchmarks/{exact_file_name}"
    bench_file_path = f"{base_path}.csv" if os.path.exists(f"{base_path}.csv") else (f"{base_path}.xlsx" if os.path.exists(f"{base_path}.xlsx") else None)
        
    if not bench_file_path:
        st.error(f"⚠️ **Benchmark Missing:** The engine could not find the backend file. Please ask your analytics team to upload `{exact_file_name}.csv` into the `benchmarks/` folder on GitHub.")
        return
        
    # ... (The rest of the code for Module 3 remains exactly the same below this!)

# ==============================================================================
# 🗺️ NAVIGATION & MAIN APP CONTROL
# ==============================================================================
st.sidebar.markdown("<h2 style='color:#002551;'>🚀 Control Panel</h2>", unsafe_allow_html=True)
pipeline_mode = st.sidebar.radio("Select Strategy Module:", ["📁 Single Campaign Matrix", "📊 Head-to-Head Variance", "🏆 Yearly Benchmark Scorecard"])
if pipeline_mode == "📁 Single Campaign Matrix": render_single_campaign_matrix()
elif pipeline_mode == "📊 Head-to-Head Variance": render_head_to_head_variance()
elif pipeline_mode == "🏆 Yearly Benchmark Scorecard": render_benchmark_scorecard()
