import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import re
import os

# Create the hidden directories on the server if they don't exist
if not os.path.exists("benchmarks"):
    os.makedirs("benchmarks")
if not os.path.exists("reference_data"):
    os.makedirs("reference_data")

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
            'run_id': get_col(['Flyer Run ID', 'Run ID', 'Campaign ID']),
            'run_name': get_col(['Flyer Description', 'Flyer Run Name', 'Campaign Name']),
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
    df['Run_ID'] = df[m['run_id']].astype(str) if m.get('run_id') else "UNKNOWN"
    df['Flyer_Description'] = df[m['run_name']].astype(str) if m.get('run_name') else df['Run_ID']
    
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
    
    def get_sort_val(x):
        s = str(x).lower()
        if 'open' in s: return -1
        if 'finish' in s or 'complete' in s: return 9999
        nums = re.findall(r'\d+', s)
        return float(nums[0]) if nums else 999

    id_col = next((c for c in df_sc.columns if 'flyer run name' in c.lower()), None)
    if not id_col:
        id_col = next((c for c in df_sc.columns if 'flyer run id' in c.lower()), None)
    if not id_col: 
        id_col = next((c for c in df_sc.columns if any(k in c.lower() for k in ['date', 'run', 'campaign', 'week', 'title'])), None)
    
    weekly_data = None
    qbr_insights = None

    if 'scroll depth' in cols_lower and 'cumulative readers' in cols_lower and 'total readers' in cols_lower:
        get_exact = lambda name: next((c for c in df_sc.columns if c.lower() == name), None)
        sd_col, pr_col, cr_col, tr_col = get_exact('scroll depth'), get_exact('pages read'), get_exact('cumulative readers'), get_exact('total readers')
        
        if pr_col: df_sc[pr_col] = pd.to_numeric(df_sc[pr_col], errors='coerce').fillna(0)
        df_sc['sort_val'] = df_sc[sd_col].apply(get_sort_val)
        
        agg = df_sc.groupby(sd_col).agg({pr_col: 'mean' if pr_col else 'first', cr_col: 'sum', tr_col: 'sum', 'sort_val': 'first'}).reset_index()
        agg['Retention'] = np.where(agg[tr_col] > 0, agg[cr_col] / agg[tr_col], 0)
        agg = agg.sort_values('sort_val')
        
        if pr_col: agg['Approx Page'] = agg[pr_col].round(1)
        else: agg['Approx Page'] = "N/A"
        agg['Milestone'] = agg[sd_col]
        
        if id_col and df_sc[id_col].nunique() > 1:
            week_agg = df_sc.groupby([id_col, sd_col]).agg({cr_col: 'sum', tr_col: 'sum', 'sort_val': 'first'}).reset_index()
            week_agg['Retention'] = np.where(week_agg[tr_col] > 0, week_agg[cr_col] / week_agg[tr_col], 0)
            weekly_data = week_agg.sort_values([id_col, 'sort_val']).rename(columns={id_col: 'Campaign/Week', sd_col: 'Milestone'})
            
            # --- THE 3-POINT QBR CALCULATION ---
            week_score = weekly_data.groupby('Campaign/Week')['Retention'].sum()
            vol_week = week_score.idxmax()
            vol_score = week_score.max()

            counts = weekly_data.groupby('Campaign/Week')['Milestone'].count()
            valid_weeks = counts[counts > 2].index
            if len(valid_weeks) == 0: valid_weeks = counts.index
            
            eff_scores = weekly_data[weekly_data['Campaign/Week'].isin(valid_weeks)].groupby('Campaign/Week')['Retention'].apply(lambda x: x.diff().mean())
            eff_week = eff_scores.idxmax() 
            eff_drop = abs(eff_scores.max()) if pd.notna(eff_scores.max()) else 0

            hl_data = weekly_data[(weekly_data['Campaign/Week'] == vol_week) & (weekly_data['Retention'] < 0.50)]
            hl_milestone = hl_data.iloc[0]['Milestone'] if not hl_data.empty else "Finished Flyer"
                
            qbr_insights = {
                'vol_week': vol_week,
                'vol_score': vol_score,
                'eff_week': eff_week,
                'eff_drop': eff_drop,
                'hl_milestone': hl_milestone
            }

    else:
        df_sc = df_sc.iloc[:, :3]
        df_sc.columns = ['Milestone', 'Readers', 'Retention']
        df_sc['Retention'] = pd.to_numeric(df_sc['Retention'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df_sc['Retention'] = np.where(df_sc['Retention'] > 1, df_sc['Retention'] / 100, df_sc['Retention'])
        df_sc['sort_val'] = df_sc['Milestone'].apply(get_sort_val)
        agg = df_sc.dropna(subset=['Retention']).sort_values('sort_val').copy()
        agg['Approx Page'] = "N/A"
        
    if period_name: agg['Period'] = period_name
    
    final_df = agg[['Milestone', 'Retention', 'Approx Page', 'Period'] if period_name else ['Milestone', 'Retention', 'Approx Page']]
    return final_df, weekly_data, qbr_insights

def generate_h2h_insight(gloA, gloB, cat_m_l1):
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
        
    if not cat_m_l1.empty:
        cat_m_l1['Efficiency'] = cat_m_l1['Alloc Variant %'] - cat_m_l1['Alloc Base %']
        top_cat = cat_m_l1.loc[cat_m_l1['Allocation Shift'].idxmax()]['L1_Category']
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
# 🗂️ MODULE 1: SINGLE CAMPAIGN BREAKDOWN
# ==============================================================================
def render_single_campaign_matrix():
    st.markdown("<div class='main-header'>Single Campaign Breakdown</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload raw exports directly to map campaign performance. Files can be processed together or independently.</div>", unsafe_allow_html=True)
    
    dl_placeholder = st.empty()
    
    col1, col2 = st.columns(2)
    with col1: merch_file = st.file_uploader("📁 Upload Merchandise File (.xlsx/.csv)", type=["xlsx", "csv"])
    with col2: scroll_file = st.file_uploader("📉 Upload Scroll Depth File (.xlsx/.csv)", type=["xlsx", "csv"])
        
    if not merch_file and not scroll_file:
        st.info("⚠️ **Waiting for data:** Please upload a Merchandise file, a Scroll Depth file, or both to begin analysis.")
        return

    pivot_top = cat_l1_agg = cat_l2_agg = cat_l3_agg = brand_agg = cr_agg = p_agg = d_agg = pd.DataFrame()
    df_sc_table = pd.DataFrame()
    weekly_scroll = pd.DataFrame()
    qbr_insights = None

    if merch_file:
        df_clean, m, header_idx = scrub_and_load_excel(merch_file)
        if df_clean is not None:
            df_prod, df_creative, global_totals = process_metrics(df_clean, m)
            merchant, run_name, run_id, date_from, date_to = extract_exact_metadata(df_clean)
            
            pivot_top = df_prod.groupby('SKU').agg({'Name': 'first', 'Page': 'first', 'Views': 'sum', 'Clicks': 'sum', 'Clips': 'sum', 'TTMs': 'sum'}).reset_index()
            pivot_top['Item CTR'] = np.where(pivot_top['Views'] > 0, pivot_top['Clicks'] / pivot_top['Views'], 0.0)
            
            def build_cat_agg(cat_col):
                c_agg = df_prod.groupby(cat_col).agg(Count=('SKU', 'count'), Views=('Views', 'sum'), Clicks=('Clicks', 'sum'), Clips=('Clips', 'sum'), TTMs=('TTMs', 'sum')).reset_index()
                c_agg['Item Allocation %'] = c_agg['Count'] / c_agg['Count'].sum() if c_agg['Count'].sum() > 0 else 0
                c_agg['Click Share %'] = c_agg['Clicks'] / c_agg['Clicks'].sum() if c_agg['Clicks'].sum() > 0 else 0
                return c_agg
            cat_l1_agg, cat_l2_agg, cat_l3_agg = build_cat_agg('L1_Category'), build_cat_agg('L2_Category'), build_cat_agg('L3_Category')
            
            brand_agg = df_prod.groupby('Brand').agg(Unique_Items=('SKU', 'nunique'), Views=('Views','sum'), Clicks=('Clicks','sum'), Clips=('Clips','sum'), TTMs=('TTMs','sum')).reset_index()
            brand_agg['Click Share %'] = brand_agg['Clicks'] / global_totals['clicks'] if global_totals['clicks'] > 0 else 0
            brand_agg['List Share %'] = brand_agg['Clips'] / global_totals['clips'] if global_totals['clips'] > 0 else 0
            brand_agg['TTM Share %'] = brand_agg['TTMs'] / global_totals['ttms'] if global_totals['ttms'] > 0 else 0
            
            if not df_creative.empty:
                cr_agg = df_creative.groupby('Name').agg(Page=('Page','max'), Views=('Views','sum'), Clicks=('Clicks','sum')).reset_index()
                cr_agg['Asset CTR'] = np.where(cr_agg['Views'] > 0, cr_agg['Clicks'] / cr_agg['Views'], 0)
                
            df_prod_bands = df_prod.copy()
            
            price_bins = [-1, 10, 25, 50, 100, 250, 500, 1000, 1500, float('inf')]
            price_labels = ["$0 - $10", "$11 - $25", "$26 - $50", "$51 - $100", "$101 - $250", "$251 - $500", "$501 - $1000", "$1001 - $1500", "$1500+"]
            
            df_prod_bands['Price_Tier'] = pd.cut(df_prod_bands['Curr_Price'], bins=price_bins, labels=price_labels)
            df_prod_bands['Discount_Tier'] = pd.cut(df_prod_bands['Discount_Pct'], bins=[-1, 0, 15, 30, 50, float('inf')], labels=["No Discount", "1% - 15%", "16% - 30%", "31% - 50%", "50%+"])
            
            p_agg = df_prod_bands.groupby('Price_Tier', observed=False).agg(Items=('SKU', 'nunique'), Clicks=('Clicks', 'sum'), Clips=('Clips', 'sum'), TTMs=('TTMs', 'sum')).reset_index()
            p_agg['Click Share %'] = p_agg['Clicks'] / global_totals['clicks'] if global_totals['clicks'] > 0 else 0
            p_agg['List Share %'] = p_agg['Clips'] / global_totals['clips'] if global_totals['clips'] > 0 else 0
            p_agg['TTM Share %'] = p_agg['TTMs'] / global_totals['ttms'] if global_totals['ttms'] > 0 else 0
            p_agg = p_agg[p_agg['Items'] > 0]

            d_agg = df_prod_bands.groupby('Discount_Tier', observed=False).agg(Items=('SKU', 'nunique'), Clicks=('Clicks', 'sum'), Clips=('Clips', 'sum'), TTMs=('TTMs', 'sum')).reset_index()
            d_agg['Click Share %'] = d_agg['Clicks'] / global_totals['clicks'] if global_totals['clicks'] > 0 else 0
            d_agg['List Share %'] = d_agg['Clips'] / global_totals['clips'] if global_totals['clips'] > 0 else 0
            d_agg['TTM Share %'] = d_agg['TTMs'] / global_totals['ttms'] if global_totals['ttms'] > 0 else 0
            d_agg = d_agg[d_agg['Items'] > 0]

    if scroll_file:
        try:
            df_sc_raw, weekly_scroll, qbr_insights = process_scroll_file(scroll_file)
            df_sc_table = df_sc_raw.copy().rename(columns={'Milestone': 'Scroll Depth', 'Retention': '% of Users Read'})
        except Exception as e:
            st.warning(f"Could not process the scroll file. Error: {str(e)}")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        wrote_any = False
        if not pivot_top.empty:
            pivot_top.sort_values(by='Item CTR', ascending=False).head(50).to_excel(writer, sheet_name='Top Items', index=False)
            cat_l1_agg.sort_values(by='Clicks', ascending=False).to_excel(writer, sheet_name='L1 Categories', index=False)
            cat_l2_agg.sort_values(by='Clicks', ascending=False).to_excel(writer, sheet_name='L2 Categories', index=False)
            cat_l3_agg.sort_values(by='Clicks', ascending=False).to_excel(writer, sheet_name='L3 Categories', index=False)
            brand_agg.sort_values(by='Clicks', ascending=False).to_excel(writer, sheet_name='Brand Momentum', index=False)
            if not cr_agg.empty: cr_agg.sort_values(by='Clicks', ascending=False).to_excel(writer, sheet_name='Creative Assets', index=False)
            p_agg.sort_values(by='Price_Tier').to_excel(writer, sheet_name='Price Bands', index=False)
            d_agg.sort_values(by='Discount_Tier').to_excel(writer, sheet_name='Discount Bands', index=False)
            wrote_any = True
        if not df_sc_table.empty:
            df_sc_table.to_excel(writer, sheet_name='Scroll Drop-off', index=False)
            wrote_any = True
        if weekly_scroll is not None and not weekly_scroll.empty:
            weekly_scroll.to_excel(writer, sheet_name='Weekly Scroll Variance', index=False)
            wrote_any = True
            
        if not wrote_any:
            pd.DataFrame({'Message': ['No data processed.']}).to_excel(writer, sheet_name='Empty Data', index=False)

    output.seek(0)
    
    dl_placeholder.download_button(
        label="⬇️ Download Dashboard Report (.xlsx)",
        data=output,
        file_name=f"Campaign_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if merch_file and df_clean is not None:
        st.info(f"📍 **ACTIVE FLIGHT RECAP:** {merchant}  |  **Flight Group:** {run_name} (ID: {run_id})  |  **Window:** {date_from} to {date_to}")
        
        cat_clicks = df_prod.groupby('L1_Category')['Clicks'].sum() if not df_prod.empty else pd.Series(dtype=float)
        top_cat = cat_clicks.idxmax() if not cat_clicks.empty else "General Merchandise"

        brand_clicks = df_prod.groupby('Brand')['Clicks'].sum() if not df_prod.empty else pd.Series(dtype=float)
        top_brand = brand_clicks.idxmax() if not brand_clicks.empty else "UNKNOWN"
        
        render_insight_box(
            f"The campaign generated **{global_totals['views']:,.0f} views** and **{global_totals['clicks']:,.0f} clicks**, achieving an overall item CTR of **{(global_totals['clicks']/global_totals['views']) if global_totals['views']>0 else 0:.2%}**.",
            f"Audience engagement was heavily concentrated, with **{top_cat}** acting as the primary traffic driver for departments, and **{top_brand}** dominating brand-level affinity.",
            f"**1.** Ensure future campaigns allocate sufficient premier page placement to {top_cat}.<br>**2.** Investigate the top 10 items by CTR and absolute Clicks to identify high-performing assets that can be repurposed in future creative."
        )
        
        # --- NEW TWO-TIERED SUMMARY DASHBOARD ---
        v_tot, cl_tot, cp_tot, t_tot = global_totals['views'], global_totals['clicks'], global_totals['clips'], global_totals['ttms']
        ctr_global_display = f"{cl_tot/v_tot:.2%}" if v_tot > 0 else "0.00%"
        
        item_v_tot = df_prod['Views'].sum() if not df_prod.empty else 0
        item_cl_tot = df_prod['Clicks'].sum() if not df_prod.empty else 0
        item_cp_tot = df_prod['Clips'].sum() if not df_prod.empty else 0
        item_t_tot = df_prod['TTMs'].sum() if not df_prod.empty else 0
        item_ctr_display = f"{item_cl_tot/item_v_tot:.2%}" if item_v_tot > 0 else "0.00%"
        
        st.markdown("<h4 style='color:#002551; margin-top:20px;'>🌐 Overall Campaign Totals (Includes Marketing Assets)</h4>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f"<div class='metric-card'><div class='metric-val'>{v_tot:,.0f}</div><div class='metric-lbl'>TOTAL VIEWS</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-val'>{cl_tot:,.0f}</div><div class='metric-lbl'>TOTAL CLICKS</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-val'>{cp_tot:,.0f}</div><div class='metric-lbl'>TOTAL ADD TO LISTS</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-val'>{t_tot:,.0f}</div><div class='metric-lbl'>TOTAL TTMS</div></div>", unsafe_allow_html=True)
        c5.markdown(f"<div class='metric-card'><div class='metric-val'>{ctr_global_display}</div><div class='metric-lbl'>GLOBAL CTR</div></div>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='color:#002551; margin-top:20px;'>🛒 Item-Specific Performance (Products Only)</h4>", unsafe_allow_html=True)
        i1, i2, i3, i4, i5 = st.columns(5)
        i1.markdown(f"<div class='metric-card'><div class='metric-val'>{item_v_tot:,.0f}</div><div class='metric-lbl'>TOTAL ITEM VIEWS</div></div>", unsafe_allow_html=True)
        i2.markdown(f"<div class='metric-card'><div class='metric-val'>{item_cl_tot:,.0f}</div><div class='metric-lbl'>ITEM CLICKS</div></div>", unsafe_allow_html=True)
        i3.markdown(f"<div class='metric-card'><div class='metric-val'>{item_cp_tot:,.0f}</div><div class='metric-lbl'>ITEM ADD TO LISTS</div></div>", unsafe_allow_html=True)
        i4.markdown(f"<div class='metric-card'><div class='metric-val'>{item_t_tot:,.0f}</div><div class='metric-lbl'>ITEM TTMS</div></div>", unsafe_allow_html=True)
        i5.markdown(f"<div class='metric-card'><div class='metric-val'>{item_ctr_display}</div><div class='metric-lbl'>ITEM CTR</div></div>", unsafe_allow_html=True)
        
        st.write("---")
        st.subheader("🏆 Top 10 Items by Total Clicks (Volume)")
        st.dataframe(pivot_top[['SKU', 'Name', 'Page', 'Views', 'Clicks', 'Clips', 'TTMs', 'Item CTR']].sort_values(by='Clicks', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)

        st.write("---")
        st.subheader("🎯 Top 10 Items by Item CTR (Efficiency)")
        st.dataframe(pivot_top[['SKU', 'Name', 'Page', 'Views', 'Clicks', 'Clips', 'TTMs', 'Item CTR']].sort_values(by='Item CTR', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)
        
        st.write("---")
        st.subheader("📊 Item Allocation vs Click Share")
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
            st.dataframe(brand_agg[['Brand', 'Unique_Items', 'Clicks', 'Click Share %', 'Clips', 'List Share %', 'TTMs', 'TTM Share %']].sort_values(by='Clicks', ascending=False).head(15).style.format({
                'Unique_Items': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 
                'Click Share %': '{:.2%}', 'List Share %': '{:.2%}', 'TTM Share %': '{:.2%}'
            }), use_container_width=True, hide_index=True)
        with c_col:
            if not df_creative.empty:
                st.dataframe(cr_agg.sort_values(by='Clicks', ascending=False).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Asset CTR': '{:.2%}'}), use_container_width=True, hide_index=True)

        st.write("---")
        st.subheader("💰 Pricing & Promotional Band Analysis")
        band_fmt = {'Items': '{:,.0f}', 'Clicks': '{:,.0f}', 'Clips': '{:,.0f}', 'TTMs': '{:,.0f}', 'Click Share %': '{:.2%}', 'List Share %': '{:.2%}', 'TTM Share %': '{:.2%}'}
        
        p_agg_sorted = p_agg.sort_values(by='Price_Tier')
        d_agg_sorted = d_agg.sort_values(by='Discount_Tier')

        c_p, c_d = st.columns(2)
        with c_p: 
            st.markdown("**Price Band Performance**")
            st.dataframe(p_agg_sorted[['Price_Tier', 'Items', 'Clicks', 'Click Share %', 'Clips', 'List Share %', 'TTMs', 'TTM Share %']].style.format(band_fmt), use_container_width=True, hide_index=True)
        with c_d: 
            st.markdown("**Discount Band Performance**")
            st.dataframe(d_agg_sorted[['Discount_Tier', 'Items', 'Clicks', 'Click Share %', 'Clips', 'List Share %', 'TTMs', 'TTM Share %']].style.format(band_fmt), use_container_width=True, hide_index=True)
            
        fig_price = px.bar(
            p_agg_sorted.melt(id_vars='Price_Tier', value_vars=['List Share %', 'TTM Share %']),
            x='Price_Tier',
            y='value',
            color='variable',
            barmode='group',
            color_discrete_sequence=['#16a34a', '#0054B7'],
            title="Price Band Impact: Add to List % vs. Transfer to Merchant %"
        )
        fig_price.update_layout(yaxis=dict(tickformat='.1%'), xaxis_title="Price Tier", yaxis_title="% of Total Share")
        st.plotly_chart(fig_price, use_container_width=True)
        
        if not p_agg_sorted.empty and global_totals['views'] > 0:
            top_list_tier = p_agg_sorted.loc[p_agg_sorted['Clips'].idxmax(), 'Price_Tier'] if p_agg_sorted['Clips'].sum() > 0 else None
            top_ttm_tier = p_agg_sorted.loc[p_agg_sorted['TTMs'].idxmax(), 'Price_Tier'] if p_agg_sorted['TTMs'].sum() > 0 else None
            
            if top_list_tier or top_ttm_tier:
                st.markdown("#### 🛍️ Hero Products in Winning Price Bands")
                col_tl, col_tt = st.columns(2)
                
                with col_tl:
                    if top_list_tier:
                        st.success(f"📋 **Top Add-to-List Tier: {top_list_tier}**")
                        top_list_items = df_prod_bands[df_prod_bands['Price_Tier'] == top_list_tier].groupby('SKU').agg({'Name': 'first', 'Curr_Price': 'first', 'Clips': 'sum'}).reset_index().sort_values('Clips', ascending=False).head(3)
                        st.dataframe(top_list_items[['SKU', 'Name', 'Curr_Price', 'Clips']].rename(columns={'Curr_Price': 'Price'}).style.format({'Price': '${:.2f}', 'Clips': '{:,.0f}'}), use_container_width=True, hide_index=True)
                        
                with col_tt:
                    if top_ttm_tier:
                        st.info(f"🛒 **Top Click-to-Buy (TTM) Tier: {top_ttm_tier}**")
                        top_ttm_items = df_prod_bands[df_prod_bands['Price_Tier'] == top_ttm_tier].groupby('SKU').agg({'Name': 'first', 'Curr_Price': 'first', 'TTMs': 'sum'}).reset_index().sort_values('TTMs', ascending=False).head(3)
                        st.dataframe(top_ttm_items[['SKU', 'Name', 'Curr_Price', 'TTMs']].rename(columns={'Curr_Price': 'Price'}).style.format({'Price': '${:.2f}', 'TTMs': '{:,.0f}'}), use_container_width=True, hide_index=True)

    if scroll_file and not df_sc_table.empty:
        st.write("---")
        st.subheader("📉 Audience Scroll Retention & Drop-off")
        
        if qbr_insights:
            st.success(f"🌟 **Insight:** The engine detected multiple campaigns/weeks. Here is how your audience engaged across the flights:")
            
            st.markdown(f"""
            **1. Total Content Consumed (Highest Volume)**
            * **Winner:** **{qbr_insights['vol_week']}**
            * **Why it won:** This flyer drove the highest absolute volume of page reads. Even if users dropped off over time, its structure generated the most total brand engagement.

            **2. Engagement Efficiency (Lowest Drop-off Velocity)**
            * **Winner:** **{qbr_insights['eff_week']}**
            * **Why it won:** This flyer was the most "gripping". It held onto its starting audience the best step-by-step, losing an average of only **{qbr_insights['eff_drop']:.1%}** of readers per scroll.

            **3. The 'Half-Life' Metric (Median Reader Depth)**
            * **Insight:** For your highest volume flyer ({qbr_insights['vol_week']}), you successfully kept the majority of your audience up until the **{qbr_insights['hl_milestone']}** mark.
            * **Why it matters:** Any products or categories placed after this 50% drop-off threshold were essentially invisible to the majority of your weekly traffic.
            """)
            
        sc_col1, sc_col2 = st.columns([1, 2])
        with sc_col1:
            st.markdown("**Global Average Retention**")
            st.dataframe(df_sc_table[['Scroll Depth', '% of Users Read', 'Approx Page']].style.format({'% of Users Read': '{:.1%}'}), use_container_width=True, hide_index=True)
        with sc_col2:
            if weekly_scroll is not None and not weekly_scroll.empty:
                fig = px.line(weekly_scroll, x='Milestone', y='Retention', color='Campaign/Week', markers=True, title="Variance by Campaign/Week")
            else:
                fig = px.line(df_sc_table, x='Scroll Depth', y='% of Users Read', markers=True, color_discrete_sequence=['#0054B7'])
            
            ordered_milestones = df_sc_table['Scroll Depth'].tolist()
            fig.update_layout(
                xaxis=dict(categoryorder='array', categoryarray=ordered_milestones),
                yaxis=dict(tickformat='.0%', range=[0,1])
            )
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 🗂️ MODULE 2: HEAD-TO-HEAD COMPARISON
# ==============================================================================
def render_head_to_head_variance():
    st.markdown("<div class='main-header'>Head-to-Head Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Compare Period A (Base Year) against Period B (Variant Year) to calculate strategic growth deltas.</div>", unsafe_allow_html=True)
    
    dl_placeholder = st.empty()
    
    colA, colB = st.columns(2)
    with colA: 
        file_A = st.file_uploader("📁 Period A: Base Merch File", type=["xlsx", "csv"])
        scroll_A = st.file_uploader("📉 Period A: Scroll Depth File [Optional]", type=["xlsx", "csv"])
    with colB: 
        file_B = st.file_uploader("📁 Period B: Variant Merch File", type=["xlsx", "csv"])
        scroll_B = st.file_uploader("📉 Period B: Scroll Depth File [Optional]", type=["xlsx", "csv"])
    
    merch_ready = file_A is not None and file_B is not None
    scroll_ready = scroll_A is not None and scroll_B is not None

    if not merch_ready and not scroll_ready:
        st.warning("⚠️ **Waiting for data:** Please upload BOTH Base and Variant files (either Merchandise or Scroll Depth) to generate a comparison.")
        return
        
    dfA_clean, mA, _ = scrub_and_load_excel(file_A) if file_A else (None, None, None)
    dfB_clean, mB, _ = scrub_and_load_excel(file_B) if file_B else (None, None, None)
    
    br_merge = cat_m_l1 = cat_m_l2 = cat_m_l3 = final_sk = new_skus = ret_skus = p_merge = d_merge = tbl_merge = pd.DataFrame()

    if merch_ready and dfA_clean is not None and dfB_clean is not None:
        dfA_prod, _, gloA = process_metrics(dfA_clean, mA)
        dfB_prod, _, gloB = process_metrics(dfB_clean, mB)
        
        _, rA, _, dA_from, dA_to = extract_exact_metadata(dfA_clean)
        _, rB, _, dB_from, dB_to = extract_exact_metadata(dfB_clean)
        
        def build_shift_matrix(col_name):
            catA = dfA_prod.groupby(col_name).agg(CntA=('SKU', 'count'), ClkA=('Clicks', 'sum')).reset_index()
            catA['Alloc Base %'] = catA['CntA'] / catA['CntA'].sum() if catA['CntA'].sum() > 0 else 0
            catB = dfB_prod.groupby(col_name).agg(CntB=('SKU', 'count'), ClkB=('Clicks', 'sum')).reset_index()
            catB['Alloc Variant %'] = catB['CntB'] / catB['CntB'].sum() if catB['CntB'].sum() > 0 else 0
            cat_m = pd.merge(catA[[col_name, 'Alloc Base %']], catB[[col_name, 'Alloc Variant %']], on=col_name, how='outer').fillna(0)
            cat_m['Allocation Shift'] = cat_m['Alloc Variant %'] - cat_m['Alloc Base %']
            return cat_m
            
        cat_m_l1 = build_shift_matrix('L1_Category')
        cat_m_l2 = build_shift_matrix('L2_Category')
        cat_m_l3 = build_shift_matrix('L3_Category')

        brA, brB = dfA_prod.groupby('Brand')[['Views', 'Clicks', 'TTMs']].sum().reset_index(), dfB_prod.groupby('Brand')[['Views', 'Clicks', 'TTMs']].sum().reset_index()
        br_merge = pd.merge(brA, brB, on='Brand', suffixes=(' Base', ' Variant'), how='outer').fillna(0)
        br_merge['TTM Growth %'] = np.where(br_merge['TTMs Base'] > 0, (br_merge['TTMs Variant'] - br_merge['TTMs Base']) / br_merge['TTMs Base'], np.where(br_merge['TTMs Variant'] > 0, 1.0, 0.0))
        br_merge['Click Growth %'] = np.where(br_merge['Clicks Base'] > 0, (br_merge['Clicks Variant'] - br_merge['Clicks Base']) / br_merge['Clicks Base'], np.where(br_merge['Clicks Variant'] > 0, 1.0, 0.0))
        
        skA = dfA_prod.groupby('SKU').agg({'Name': 'first', 'Views': 'sum', 'Clicks': 'sum', 'Curr_Price': 'mean'}).reset_index()
        skB = dfB_prod.groupby('SKU').agg({'Name': 'first', 'Views': 'sum', 'Clicks': 'sum', 'Curr_Price': 'mean'}).reset_index()
        sk_m = pd.merge(skA, skB, on='SKU', suffixes=(' Base', ' Variant'), how='inner')
        if not sk_m.empty:
            sk_m['CTR Base'] = np.where(sk_m['Views Base'] > 0, sk_m['Clicks Base'] / sk_m['Views Base'], 0)
            sk_m['CTR Variant'] = np.where(sk_m['Views Variant'] > 0, sk_m['Clicks Variant'] / sk_m['Views Variant'], 0)
            sk_m['CTR Shift'] = sk_m['CTR Variant'] - sk_m['CTR Base']
            sk_m['Price Shift'] = sk_m['Curr_Price Variant'] - sk_m['Curr_Price Base']
            final_sk = sk_m[['SKU', 'Name Variant', 'Views Variant', 'Clicks Variant', 'CTR Base', 'CTR Variant', 'CTR Shift', 'Price Shift']].copy()
            final_sk.rename(columns={'Name Variant': 'Name'}, inplace=True)
            
        new_skus = dfB_prod[~dfB_prod['SKU'].isin(dfA_prod['SKU'])].groupby('SKU').agg({'Name': 'first', 'Views': 'sum', 'Clicks': 'sum', 'TTMs': 'sum'}).reset_index()
        if not new_skus.empty: new_skus['Item CTR'] = np.where(new_skus['Views'] > 0, new_skus['Clicks'] / new_skus['Views'], 0)
        
        ret_skus = dfA_prod[~dfA_prod['SKU'].isin(dfB_prod['SKU'])].groupby('SKU').agg({'Name': 'first', 'Views': 'sum', 'Clicks': 'sum', 'TTMs': 'sum'}).reset_index()
        if not ret_skus.empty: ret_skus['Item CTR'] = np.where(ret_skus['Views'] > 0, ret_skus['Clicks'] / ret_skus['Views'], 0)

        for d in [dfA_prod, dfB_prod]: 
            price_bins = [-1, 10, 25, 50, 100, 250, 500, 1000, 1500, float('inf')]
            price_labels = ["$0 - $10", "$11 - $25", "$26 - $50", "$51 - $100", "$101 - $250", "$251 - $500", "$501 - $1000", "$1001 - $1500", "$1500+"]
            
            d['Price_Tier'] = pd.cut(d['Curr_Price'], bins=price_bins, labels=price_labels)
            d['Discount_Tier'] = pd.cut(d['Discount_Pct'], bins=[-1, 0, 15, 30, 50, float('inf')], labels=["No Discount", "1% - 15%", "16% - 30%", "31% - 50%", "50%+"])
        
        pA, pB = dfA_prod.groupby('Price_Tier', observed=False)['Clicks'].sum().reset_index().rename(columns={'Clicks': 'Base Clicks'}), dfB_prod.groupby('Price_Tier', observed=False)['Clicks'].sum().reset_index().rename(columns={'Clicks': 'Variant Clicks'})
        p_merge = pd.merge(pA, pB, on='Price_Tier').fillna(0)
        p_merge['Click Share Shift'] = (p_merge['Variant Clicks'] / p_merge['Variant Clicks'].sum()) - (p_merge['Base Clicks'] / p_merge['Base Clicks'].sum())
        p_merge = p_merge[(p_merge['Base Clicks'] > 0) | (p_merge['Variant Clicks'] > 0)]
        
        dA, dB = dfA_prod.groupby('Discount_Tier', observed=False)['Clicks'].sum().reset_index().rename(columns={'Clicks': 'Base Clicks'}), dfB_prod.groupby('Discount_Tier', observed=False)['Clicks'].sum().reset_index().rename(columns={'Clicks': 'Variant Clicks'})
        d_merge = pd.merge(dA, dB, on='Discount_Tier').fillna(0)
        d_merge['Click Share Shift'] = (d_merge['Variant Clicks'] / d_merge['Variant Clicks'].sum()) - (d_merge['Base Clicks'] / d_merge['Base Clicks'].sum())
        d_merge = d_merge[(d_merge['Base Clicks'] > 0) | (d_merge['Variant Clicks'] > 0)]

    if scroll_ready:
        try:
            df_scA, _, _ = process_scroll_file(scroll_A, 'Base Year')
            df_scB, _, _ = process_scroll_file(scroll_B, 'Variant Year')
            tbl_merge = pd.merge(df_scA[['Milestone', 'Approx Page', 'Retention']].rename(columns={'Retention': 'Base % Read', 'Approx Page': 'Base Page'}), df_scB[['Milestone', 'Approx Page', 'Retention']].rename(columns={'Retention': 'Variant % Read', 'Approx Page': 'Variant Page'}), on='Milestone', how='outer').rename(columns={'Milestone': 'Scroll Depth'})
            tbl_merge['Approx Page'] = tbl_merge['Variant Page'].combine_first(tbl_merge['Base Page'])
        except: pass

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        wrote_any = False
        if merch_ready:
            br_merge.to_excel(writer, sheet_name='Brand Momentum', index=False)
            cat_m_l1.to_excel(writer, sheet_name='L1 Category Shifts', index=False)
            cat_m_l2.to_excel(writer, sheet_name='L2 Category Shifts', index=False)
            cat_m_l3.to_excel(writer, sheet_name='L3 Category Shifts', index=False)
            if not final_sk.empty: final_sk.to_excel(writer, sheet_name='Shared SKUs Delta', index=False)
            if not new_skus.empty: new_skus.to_excel(writer, sheet_name='New SKUs', index=False)
            if not ret_skus.empty: ret_skus.to_excel(writer, sheet_name='Retired SKUs', index=False)
            p_merge.sort_values(by='Price_Tier').to_excel(writer, sheet_name='Price Shifts', index=False)
            d_merge.sort_values(by='Discount_Tier').to_excel(writer, sheet_name='Discount Shifts', index=False)
            wrote_any = True
        if not tbl_merge.empty: 
            tbl_merge.to_excel(writer, sheet_name='Scroll Shifts', index=False)
            wrote_any = True
            
        if not wrote_any:
            pd.DataFrame({'Message': ['No data processed.']}).to_excel(writer, sheet_name='Empty Data', index=False)
        
    output.seek(0)
    dl_placeholder.download_button(
        label="⬇️ Download H2H Comparison Report (.xlsx)",
        data=output,
        file_name=f"H2H_Comparison_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if merch_ready:
        st.info(f"⚖️ **COMPARING:** {rA} ({dA_from} to {dA_to}) **VERSUS** {rB} ({dB_from} to {dB_to})")
        w, sw, nw = generate_h2h_insight(gloA, gloB, cat_m_l1)
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
        st.dataframe(br_merge[['Brand', 'TTMs Base', 'TTMs Variant', 'TTM Growth %', 'Clicks Base', 'Clicks Variant', 'Click Growth %']].sort_values(by='TTM Growth %', ascending=False).style.format({'TTMs Base': '{:,.0f}', 'TTMs Variant': '{:,.0f}', 'Clicks Base': '{:,.0f}', 'Clicks Variant': '{:,.0f}', 'TTM Growth %': '{:+.1%}', 'Click Growth %': '{:+.1%}'}), use_container_width=True, hide_index=True)

        st.write("---")
        st.subheader("📊 Slot 3: Category Share Shifts")
        tab_h2h_l1, tab_h2h_l2, tab_h2h_l3 = st.tabs(["L1 Primary Category Shifts", "L2 Subcategory Shifts", "L3 Sub-subcategory Shifts"])
        with tab_h2h_l1: st.dataframe(cat_m_l1.sort_values(by='Allocation Shift', ascending=False).style.format({'Alloc Base %': '{:.1%}', 'Alloc Variant %': '{:.1%}', 'Allocation Shift': '{:+.2%} pts'}), use_container_width=True, hide_index=True)
        with tab_h2h_l2: st.dataframe(cat_m_l2.sort_values(by='Allocation Shift', ascending=False).style.format({'Alloc Base %': '{:.1%}', 'Alloc Variant %': '{:.1%}', 'Allocation Shift': '{:+.2%} pts'}), use_container_width=True, hide_index=True)
        with tab_h2h_l3: st.dataframe(cat_m_l3.sort_values(by='Allocation Shift', ascending=False).style.format({'Alloc Base %': '{:.1%}', 'Alloc Variant %': '{:.1%}', 'Allocation Shift': '{:+.2%} pts'}), use_container_width=True, hide_index=True)

        st.write("---")
        st.subheader("🏆 Slot 4: Shared SKU Micro-Delta")
        st.markdown("<small>Isolates items that appeared in BOTH campaigns to measure direct performance changes.</small>", unsafe_allow_html=True)
        if not final_sk.empty:
            st.dataframe(final_sk.sort_values(by='CTR Shift', ascending=False).style.format({'Views Variant': '{:,.0f}', 'Clicks Variant': '{:,.0f}', 'CTR Base': '{:.2%}', 'CTR Variant': '{:.2%}', 'CTR Shift': '{:+.2%} pts', 'Price Shift': '${:+.2f}'}), use_container_width=True, hide_index=True)
        else:
            st.info("No shared SKUs detected between these two flights.")
            
        st.write("---")
        st.subheader("🔄 Slot 5: YoY Assortment Turnover")
        col_new, col_ret = st.columns(2)
        with col_new:
            st.markdown("**Top New Items**")
            if not new_skus.empty: st.dataframe(new_skus.sort_values(by='Clicks', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)
            else: st.caption("No new items.")
        with col_ret:
            st.markdown("**Top Retired Items**")
            if not ret_skus.empty: st.dataframe(ret_skus.sort_values(by='Clicks', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)
            else: st.caption("No items retired.")

        st.write("---")
        st.subheader("💰 Slot 6: YoY Pricing & Promotional Shift")
        
        p_merge_sorted = p_merge.sort_values(by='Price_Tier')
        d_merge_sorted = d_merge.sort_values(by='Discount_Tier')

        c_p, c_d = st.columns(2)
        with c_p: 
            st.markdown("**YoY Price Band Shifts**")
            st.dataframe(p_merge_sorted.style.format({'Base Clicks': '{:,.0f}', 'Variant Clicks': '{:,.0f}', 'Click Share Shift': '{:+.2%}'}), use_container_width=True, hide_index=True)
        with c_d: 
            st.markdown("**YoY Discount Band Shifts**")
            st.dataframe(d_merge_sorted.style.format({'Base Clicks': '{:,.0f}', 'Variant Clicks': '{:,.0f}', 'Click Share Shift': '{:+.2%}'}), use_container_width=True, hide_index=True)

        if not dfB_prod.empty and gloB['views'] > 0:
            pB_full = dfB_prod.groupby('Price_Tier', observed=False).agg(Clips=('Clips', 'sum'), TTMs=('TTMs', 'sum')).reset_index()
            top_list_tier_B = pB_full.loc[pB_full['Clips'].idxmax(), 'Price_Tier'] if pB_full['Clips'].sum() > 0 else None
            top_ttm_tier_B = pB_full.loc[pB_full['TTMs'].idxmax(), 'Price_Tier'] if pB_full['TTMs'].sum() > 0 else None
            
            if top_list_tier_B or top_ttm_tier_B:
                st.markdown("#### 🛍️ Variant Year: Hero Products in Top Price Bands")
                st.markdown(f"<small>These specific items are driving the highest engagement in your current ({rB}) campaign.</small>", unsafe_allow_html=True)
                col_tl, col_tt = st.columns(2)
                
                with col_tl:
                    if top_list_tier_B:
                        st.success(f"📋 **Top Add-to-List Tier: {top_list_tier_B}**")
                        top_list_items = dfB_prod[dfB_prod['Price_Tier'] == top_list_tier_B].groupby('SKU').agg({'Name': 'first', 'Curr_Price': 'first', 'Clips': 'sum'}).reset_index().sort_values('Clips', ascending=False).head(3)
                        st.dataframe(top_list_items[['SKU', 'Name', 'Curr_Price', 'Clips']].rename(columns={'Curr_Price': 'Price'}).style.format({'Price': '${:.2f}', 'Clips': '{:,.0f}'}), use_container_width=True, hide_index=True)
                        
                with col_tt:
                    if top_ttm_tier_B:
                        st.info(f"🛒 **Top Click-to-Buy (TTM) Tier: {top_ttm_tier_B}**")
                        top_ttm_items = dfB_prod[dfB_prod['Price_Tier'] == top_ttm_tier_B].groupby('SKU').agg({'Name': 'first', 'Curr_Price': 'first', 'TTMs': 'sum'}).reset_index().sort_values('TTMs', ascending=False).head(3)
                        st.dataframe(top_ttm_items[['SKU', 'Name', 'Curr_Price', 'TTMs']].rename(columns={'Curr_Price': 'Price'}).style.format({'Price': '${:.2f}', 'TTMs': '{:,.0f}'}), use_container_width=True, hide_index=True)

    if not tbl_merge.empty:
        st.write("---")
        st.subheader("📉 YoY Audience Scroll Retention")
        sc_col1, sc_col2 = st.columns([1, 2])
        with sc_col1:
            st.dataframe(tbl_merge[['Scroll Depth', 'Base % Read', 'Variant % Read', 'Approx Page']].style.format({'Base % Read': '{:.1%}', 'Variant % Read': '{:.1%}'}), use_container_width=True, hide_index=True)
        with sc_col2:
            df_scA = pd.DataFrame({'Milestone': tbl_merge['Scroll Depth'], 'Retention': tbl_merge['Base % Read'], 'Period': 'Base Year'})
            df_scB = pd.DataFrame({'Milestone': tbl_merge['Scroll Depth'], 'Retention': tbl_merge['Variant % Read'], 'Period': 'Variant Year'})
            fig = px.line(pd.concat([df_scA, df_scB]), x='Milestone', y='Retention', color='Period', markers=True, color_discrete_sequence=['#475569', '#0054B7'], labels={'Milestone': 'Scroll Depth', 'Retention': '% of Users Read'})
            
            ordered_milestones = tbl_merge['Scroll Depth'].tolist()
            fig.update_layout(
                xaxis=dict(categoryorder='array', categoryarray=ordered_milestones),
                yaxis=dict(tickformat='.0%', range=[0,1])
            )
            st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 🧰 MODULE 4: TAYLOR'S WORKSPACE (REGIONAL CTR ENGINE)
# ==============================================================================
def render_taylors_workspace():
    st.markdown("<div class='main-header'>🧰 Taylor's Regional CTR Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload your Merch Metrics and FSA Zone file(s) to instantly join and calculate regional performance. The USPS reference is loaded automatically from the server. No VLOOKUPs required.</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1: merch_file = st.file_uploader("1️⃣ Upload Merchandise Metrics", type=["xlsx", "csv"])
    with col2: fsa_files = st.file_uploader("2️⃣ Upload FSA Zone Reports (Multiple Allowed)", type=["xlsx", "csv"], accept_multiple_files=True)
    
    # Path to the hidden USPS reference file (Supports CSV or XLSX)
    usps_path_xlsx = "reference_data/usps_reference.xlsx"
    usps_path_csv = "reference_data/usps_reference.csv"
    
    usps_path = None
    if os.path.exists(usps_path_csv):
        usps_path = usps_path_csv
    elif os.path.exists(usps_path_xlsx):
        usps_path = usps_path_xlsx
        
    if not usps_path:
        st.error("⚠️ **System Missing File:** Please ask your admin to place the `usps_reference.xlsx` (or `.csv`) file inside the `reference_data/` folder on the server.")
        return

    if not (merch_file and fsa_files and len(fsa_files) > 0):
        st.info("⚠️ **Awaiting Data:** Please upload your Merch file and at least one FSA file to run the pipeline.")
        return

    with st.spinner("Executing the automated pipeline natively..."):
        # 1. Load the RAW Merch Data first
        df_clean, m, _ = scrub_and_load_excel(merch_file)
        if df_clean is None: return
        
        if m.get('sku') and m['sku'] in df_clean.columns:
            raw_sku_col = m['sku']
            sku_series = df_clean[raw_sku_col].astype(str).str.strip().str.lower()
            blank_mask = df_clean[raw_sku_col].isna() | sku_series.isin(['nan', 'none', 'null', 'unknown', '0', '0.0', ''])
            df_clean = df_clean[blank_mask].copy()
            
            if df_clean.empty:
                st.error("⚠️ No products found with a blank SKU. Taylor's Workspace is configured to evaluate Non-SKU items.")
                return

        df_prod, _, _ = process_metrics(df_clean, m)
        valid_display_types = ['ITEM', 'PRODUCT']
        df_prod = df_prod[df_prod['Display_Type'].astype(str).str.upper().isin(valid_display_types)].copy()

        if df_prod.empty:
            st.error("⚠️ The engine processed the file but found zero Blank-SKU rows categorized as 'ITEM' or 'PRODUCT'.")
            return

        # 2. Load Generic Files (FSA and USPS)
        def load_generic(f):
            file_bytes = f.read()
            if f.name.lower().endswith('.csv'): 
                df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
            else:
                df = pd.read_excel(io.BytesIO(file_bytes))
            df.columns = [str(c).strip() for c in df.columns]
            return df.loc[:, ~df.columns.duplicated()]

        df_fsa = pd.concat([load_generic(f) for f in fsa_files], ignore_index=True)
        df_fsa = df_fsa.loc[:, ~df_fsa.columns.duplicated()]
        
        if usps_path.endswith('.csv'):
            df_usps = pd.read_csv(usps_path, dtype=str, low_memory=False) 
        else:
            df_usps = pd.read_excel(usps_path, dtype=str)
            
        df_usps.columns = [str(c).strip() for c in df_usps.columns]
        df_usps = df_usps.loc[:, ~df_usps.columns.duplicated()]
        
        # 3. SMARTER Column Identification Helper
        def get_col_fuzzy_strict(df, keywords, exclude_cols=None):
            exclude_cols = exclude_cols or []
            # Exact Match First
            for k in keywords:
                for col in df.columns:
                    if col not in exclude_cols and str(col).strip().lower() == k: return col
            # Contains Match Second
            for k in keywords:
                for col in df.columns:
                    if col not in exclude_cols and k in str(col).lower(): return col
            return "UNKNOWN"

        # Explicitly target 'pricing zone name' before anything else
        fsa_desc_col = get_col_fuzzy_strict(df_fsa, ['pricing zone name', 'description', 'flyer', 'campaign', 'name', 'zone'])
        if fsa_desc_col == "UNKNOWN":
            fsa_desc_col = df_fsa.columns[0] # Fallback to first column if totally lost

        # Find and exclude the ID column so we don't accidentally grab it later
        exclude_cols = [fsa_desc_col]
        id_col = get_col_fuzzy_strict(df_fsa, ['pricing zone id', 'id'])
        if id_col != "UNKNOWN": exclude_cols.append(id_col)

        fsa_zip_col = get_col_fuzzy_strict(df_fsa, ['fsa', 'zip', 'postal'], exclude_cols=exclude_cols)
        if fsa_zip_col == "UNKNOWN":
            fsa_zip_col = [c for c in df_fsa.columns if c not in exclude_cols][0]

        usps_zip_col = get_col_fuzzy_strict(df_usps, ['fsa', 'zip', 'postal'])
        if usps_zip_col == "UNKNOWN": usps_zip_col = df_usps.columns[0]
        
        usps_state_col = get_col_fuzzy_strict(df_usps, ['state', 'province', 'st', 'region', 'terr'], exclude_cols=[usps_zip_col])
        if usps_state_col == "UNKNOWN": usps_state_col = [c for c in df_usps.columns if c != usps_zip_col][0]

        # --- ARMORED KEY CLEANING & ZIP CODE PADDING ---
        def safe_pad_zip(z):
            z = str(z).strip().upper().replace(' ', '').replace('.0', '')
            if z == 'NAN' or z == 'NONE': return 'UNKNOWN'
            if z.isdigit() and len(z) < 5:
                return z.zfill(5)
            return z

        df_fsa[fsa_zip_col] = df_fsa[fsa_zip_col].apply(safe_pad_zip)
        df_usps[usps_zip_col] = df_usps[usps_zip_col].apply(safe_pad_zip)

        def aggressive_key_clean(s):
            cleaned = re.sub(r'[^A-Z0-9]', '', str(s).upper())
            # Strip the word "ZONE" if it exists, so "ZONE3" perfectly matches "3"
            if cleaned.startswith("ZONE") and len(cleaned) > 4:
                cleaned = cleaned.replace("ZONE", "")
            return cleaned
            
        df_prod['Flyer_Join_Key'] = df_prod['Flyer_Description'].apply(aggressive_key_clean)
        df_fsa['FSA_Join_Key'] = df_fsa[fsa_desc_col].apply(aggressive_key_clean)

        df_usps_unique = df_usps[[usps_zip_col, usps_state_col]].drop_duplicates(subset=[usps_zip_col])
        campaign_zips = df_fsa.merge(df_usps_unique, left_on=fsa_zip_col, right_on=usps_zip_col, how='inner')
        campaign_states = campaign_zips[['FSA_Join_Key', usps_state_col]].drop_duplicates()
        
        def assign_custom_region(state_code):
            st_clean = str(state_code).strip().upper()
            
            # East Region (+ DC, WV, IN, NC)
            if st_clean in ['DE', 'MD', 'NJ', 'OH', 'PA', 'VA', 'DC', 'WV', 'IN', 'NC', 
                            'DELAWARE', 'MARYLAND', 'NEW JERSEY', 'OHIO', 'PENNSYLVANIA', 'VIRGINIA', 'DISTRICT OF COLUMBIA', 'WEST VIRGINIA', 'INDIANA', 'NORTH CAROLINA']: 
                return 'East'
                
            # West Region (+ AZ)
            if st_clean in ['CA', 'AZ', 'CALIFORNIA', 'ARIZONA']: 
                return 'West'
                
            # Northwest Region (+ MT)
            if st_clean in ['ID', 'OR', 'WA', 'MT', 'IDAHO', 'OREGON', 'WASHINGTON', 'MONTANA']: 
                return 'Northwest'
                
            # Nevada Region
            if st_clean in ['LV', 'NV', 'NEVADA', 'LAS VEGAS']: 
                return 'Nevada'
                
            # Any total outliers will be grouped cleanly here instead of showing as isolated states
            return 'Other'
            
        campaign_states['Region'] = campaign_states[usps_state_col].apply(assign_custom_region)
        
        # 🚨 Fix: Keep only the primary region for zones that physically cross state borders to prevent metric duplication
        campaign_region_map = campaign_states[['FSA_Join_Key', 'Region']].drop_duplicates(subset=['FSA_Join_Key'], keep='first')
        
        df_prod = df_prod.merge(campaign_region_map, left_on='Flyer_Join_Key', right_on='FSA_Join_Key', how='left')
        
        # 🚨 Fix: Upgraded Smarter Matching Logic for missed connections
        unmatched_mask = df_prod['Region'].isna()
        if unmatched_mask.any():
            fallback_list = campaign_region_map.dropna(subset=['FSA_Join_Key', 'Region']).values.tolist()
            def smarter_match(m_key):
                m_str = str(m_key).strip().lower()
                if not m_str or m_str in ['nan', 'none']: return 'Other'
                
                # 1. Exact Mathematical Number Matching
                m_nums = set([str(int(n)) for n in re.findall(r'\d+', m_str)])
                for f_key, reg in fallback_list:
                    f_nums = set([str(int(n)) for n in re.findall(r'\d+', str(f_key))])
                    if f_nums and f_nums.issubset(m_nums):
                        return reg
                        
                # 2. Exact Word Boundary Matching (for text keys)
                m_words = set(re.findall(r'\b\w+\b', m_str))
                for f_key, reg in fallback_list:
                    f_words = set(re.findall(r'\b\w+\b', str(f_key).lower()))
                    if f_words and f_words.issubset(m_words):
                        return reg
                        
                # 3. Final Fallback (only for long strings to prevent small number collision)
                for f_key, reg in fallback_list:
                    f_clean = str(f_key).strip().lower()
                    if len(f_clean) >= 4 and f_clean in m_str:
                        return reg

                return 'Other'
                
            # Use original flyer description with spaces intact for accurate word/number extraction
            df_prod.loc[unmatched_mask, 'Region'] = df_prod.loc[unmatched_mask, 'Flyer_Description'].apply(smarter_match)
                        
        df_prod['Region'] = df_prod['Region'].fillna('Other')
        
        def taylor_name_scrubber(text):
            text = str(text).lower()
            text = re.sub(r'\(.*?\)', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\b\d+(\.\d+)?\s*(g|kg|ml|l|oz|lb|pk|pack|ea|ct)\b', '', text)
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            return re.sub(r'\s+', ' ', text).strip().title()
            
        df_prod['Clean_Name'] = df_prod['Name'].apply(taylor_name_scrubber)
        
        def get_taylor_cat(name, l1, l2):
            text = f"{name} {l1} {l2}".lower()
            if any(w in text for w in ['wine', 'beer', 'spirit', 'liquor', 'vodka', 'whiskey', 'tequila', 'ipa', 'alcohol']): return 'Alcohol'
            if 'bacon' in text: return 'Bacon'
            if any(w in text for w in ['ice cream', 'gelato', 'sorbet', 'popsicle']): return 'Ice Cream'
            if any(w in text for w in ['butter', 'margarine']): return 'Butter'
            if any(w in text for w in ['cheese', 'cheddar', 'mozzarella', 'provolone', 'gouda', 'brie', 'feta', 'parmesan']): return 'Cheese'
            if any(w in text for w in ['milk', 'yogurt', 'sour cream', 'cream cheese', 'cottage cheese', 'dairy']): return 'Dairy'
            if 'egg' in text and 'leg' not in text: return 'Eggs'
            if any(w in text for w in ['dog', 'cat', 'pet', 'kibble', 'treat']): return 'Pet'
            if any(w in text for w in ['frozen', 'pizza (frozen)']): return 'Frozen'
            if any(w in text for w in ['bakery', 'bread', 'muffin', 'bagel', 'croissant', 'fresh baked', 'bun', 'roll', 'tart', 'pastry']): return 'Bakery'
            if any(w in text for w in ['fish', 'salmon', 'shrimp', 'crab', 'seafood', 'tuna', 'lobster', 'cod', 'tilapia']): return 'Seafood'
            if any(w in text for w in ['deli', 'cold cut', 'salami', 'prosciutto', 'sliced ham', 'sliced turkey']): return 'Deli'
            if any(w in text for w in ['beef', 'chicken', 'pork', 'steak', 'ground beef', 'poultry', 'meat', 'roast', 'breast', 'thigh']): return 'Fresh Meat'
            if any(w in text for w in ['produce', 'fresh fruit', 'fresh veg', 'apple', 'banana', 'lettuce', 'tomato', 'potato', 'onion', 'berry', 'grapes']): return 'Produce'
            if any(w in text for w in ['water', 'soda', 'pop', 'juice', 'coke', 'pepsi', 'coffee', 'tea', 'beverage', 'drink']): return 'Beverages'
            if any(w in text for w in ['detergent', 'cleaner', 'paper towel', 'toilet', 'soap', 'trash bag', 'home']): return 'Home'
            return 'Grocery'
            
        df_prod['cat_m'] = df_prod.apply(lambda row: get_taylor_cat(row['Clean_Name'], row['L1_Category'], row['L2_Category']), axis=1)

    # 🚨 DIAGNOSTIC DASHBOARD INJECTION 🚨
    with st.expander("🛠️ PIPELINE DIAGNOSTICS (Click to expand)"):
        st.markdown("**1. What Columns Did the Engine Grab?**")
        st.write(f"- Merch Flyer Column: `{m['run_name']}`")
        st.write(f"- FSA Flyer Column: `{fsa_desc_col}`")
        st.write(f"- FSA ZIP Column: `{fsa_zip_col}`")
        st.write(f"- USPS ZIP Column: `{usps_zip_col}`")
        st.write(f"- USPS State Column: `{usps_state_col}`")
        
        st.markdown("**2. ZIP Code Handshake Test**")
        st.write(f"Total Matches found between FSA file and USPS File: **{len(campaign_zips)}**")
        if len(campaign_zips) == 0:
            st.error("🚨 FAILURE: The ZIP codes in the FSA file do not match anything in the USPS file.")
            st.write("First 5 FSA ZIPs:", df_fsa[fsa_zip_col].head(5).tolist())
            st.write("First 5 USPS ZIPs:", df_usps[usps_zip_col].head(5).tolist())

        st.markdown("**3. Flyer Name Handshake Test**")
        st.write("First 5 Flyer Names in Merch File:", df_prod['Flyer_Join_Key'].head(5).tolist())
        st.write("First 5 Flyer Names in FSA File:", df_fsa['FSA_Join_Key'].head(5).tolist())

    st.success("✅ **Data Merged Successfully!** Blank SKUs filtered to actual ITEMS, Product names unified, and regions matched.")
    
    st.write("---")
    st.subheader("📊 Top Categories by Shopper Engagement")
    cat_agg = df_prod.groupby('cat_m').agg({'Views': 'sum', 'Clicks': 'sum'}).reset_index()
    cat_agg['Category CTR'] = np.where(cat_agg['Views'] > 0, cat_agg['Clicks'] / cat_agg['Views'], 0)
    cat_agg = cat_agg.sort_values(by='Category CTR', ascending=False).head(15)
    
    fig_cat = px.bar(cat_agg, x='cat_m', y='Category CTR', title='Top Categories by Shopper Engagement', color_discrete_sequence=['#0054B7'])
    fig_cat.update_layout(yaxis=dict(tickformat='.2%'), xaxis_title="Product Category (cat_m)", yaxis_title="Category CTR")
    st.plotly_chart(fig_cat, use_container_width=True)

    st.write("---")
    st.subheader("🏆 Top 10 Items - Shopper Interest by Item CTR")
    top_items = df_prod.groupby('Clean_Name').agg({'cat_m': 'first', 'Views': 'sum', 'Clicks': 'sum'}).reset_index()
    top_items.rename(columns={'Clean_Name': 'Product Name'}, inplace=True)
    top_items['Item CTR'] = np.where(top_items['Views'] > 0, top_items['Clicks'] / top_items['Views'], 0)
    st.dataframe(top_items.sort_values(by='Item CTR', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)

    st.write("---")
    st.subheader("🏆 Top 10 Items - Shopper Interest by Clicks")
    st.dataframe(top_items.sort_values(by='Clicks', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)

    st.write("---")
    st.subheader("🗺️ Category Engagement by Region")
    reg_cat_agg = df_prod.groupby(['cat_m', 'Region']).agg({'Views': 'sum', 'Clicks': 'sum'}).reset_index()
    reg_cat_agg['CTR'] = np.where(reg_cat_agg['Views'] > 0, reg_cat_agg['Clicks'] / reg_cat_agg['Views'], 0)
    
    if not reg_cat_agg.empty:
        pivot_reg = reg_cat_agg.pivot(index='cat_m', columns='Region', values='CTR').fillna(0)
        st.dataframe(pivot_reg.style.format('{:.2%}'), use_container_width=True)
    else:
        st.info("No regional category trends found.")

    st.write("---")
    st.subheader("📍 Top 5 Items by Region & Item CTR")
    unique_regions = [r for r in df_prod['Region'].unique() if pd.notna(r) and r != 'Other']
    
    if unique_regions:
        tab_reg = st.tabs(list(unique_regions))
        for i, r in enumerate(unique_regions):
            with tab_reg[i]:
                reg_items = df_prod[df_prod['Region'] == r].groupby('Clean_Name').agg({'Views': 'sum', 'Clicks': 'sum'}).reset_index()
                reg_items.rename(columns={'Clean_Name': 'Product Name'}, inplace=True)
                reg_items['Item CTR'] = np.where(reg_items['Views'] > 0, reg_items['Clicks'] / reg_items['Views'], 0)
                reg_items = reg_items.sort_values(by=['Item CTR', 'Clicks'], ascending=[False, False]).head(5)
                st.dataframe(reg_items.style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)
    else:
        st.info("No localized regional items captured.")

# ==============================================================================
# 🏆 MODULE 3: INDUSTRY BENCHMARKS
# ==============================================================================
def render_benchmark_scorecard():
    st.markdown("<div class='main-header'>🏆 Industry Benchmarks (DNU - IN DEV)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Compare a client's current flight directly against a historical industry baseline, aligned by season.</div>", unsafe_allow_html=True)
    
    dl_placeholder = st.empty()

    colA, colB = st.columns(2)
    with colA:
        client_file = st.file_uploader("📁 Upload Client File (Current Campaign)", type=["xlsx", "csv"])
    with colB:
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
        
    exact_file_name = benchmark_map[selected_option]
    base_path = f"benchmarks/{exact_file_name}"
    bench_file_path = f"{base_path}.csv" if os.path.exists(f"{base_path}.csv") else (f"{base_path}.xlsx" if os.path.exists(f"{base_path}.xlsx") else None)
        
    if not bench_file_path:
        st.error(f"⚠️ **Benchmark Missing:** The engine could not find the backend file. Please ask your analytics team to upload `{exact_file_name}.csv` into the `benchmarks/` folder on GitHub.")
        return

    df_client_clean, m_client, _ = scrub_and_load_excel(client_file)
    if df_client_clean is None: return
    client_prod, client_creative, client_glo = process_metrics(df_client_clean, m_client)
    _, _, _, date_from, date_to = extract_exact_metadata(df_client_clean)
    
    client_start_dt = pd.to_datetime(date_from, errors='coerce') if date_from != "N/A" else pd.NaT
    client_end_dt = pd.to_datetime(date_to, errors='coerce') if date_to != "N/A" else pd.NaT

    with st.spinner(f"Crunching the massive backend data dump..."):
        df_bench_clean, m_bench, _ = scrub_and_load_excel(bench_file_path, is_local_path=True)
        if df_bench_clean is None: return
        bench_prod, bench_creative, bench_glo = process_metrics(df_bench_clean, m_bench)

    if pd.notna(client_start_dt) and pd.notna(client_end_dt) and 'Date' in bench_prod.columns and bench_prod['Date'].notna().any():
        start_md, end_md = client_start_dt.strftime('%m-%d'), client_end_dt.strftime('%m-%d')
        st.info(f"📅 **Seasonal Alignment Active:** Filtering historical data to exactly match the **{client_start_dt.strftime('%b %d')} to {client_end_dt.strftime('%b %d')}** window.")
        
        def filter_by_season(df):
            if df.empty or 'Date' not in df.columns: return df
            md = df['Date'].dt.strftime('%m-%d')
            mask = (md >= start_md) & (md <= end_md) if start_md <= end_md else (md >= start_md) | (md <= end_md)
            return df[mask]
            
        bench_prod = filter_by_season(bench_prod)
        bench_creative = filter_by_season(bench_creative)

    c_item_ctr = client_prod['Clicks'].sum() / client_prod['Views'].sum() if client_prod['Views'].sum() > 0 else 0
    b_item_ctr = bench_prod['Clicks'].sum() / bench_prod['Views'].sum() if bench_prod['Views'].sum() > 0 else 0
    
    c_bnr_ctr = client_creative['Clicks'].sum() / client_creative['Views'].sum() if client_creative['Views'].sum() > 0 else 0
    b_bnr_ctr = bench_creative['Clicks'].sum() / bench_creative['Views'].sum() if bench_creative['Views'].sum() > 0 else 0

    def get_avg_pages(df):
        if df.empty: return 0
        valid = df[df['Run_ID'] != "UNKNOWN"]
        if not valid.empty:
            return valid.groupby('Run_ID')['Page'].max().mean()
        return df['Page'].max()

    c_pages = get_avg_pages(client_prod)
    b_pages = get_avg_pages(bench_prod)
    
    scorecard_df = pd.DataFrame({
        "Metric": ["Avg. Item CTR", "Marketing Banner CTR", "Avg. Flyer Length (Pages)"],
        "Client Performance": [f"{c_item_ctr:.2%}", f"{c_bnr_ctr:.2%}", f"{c_pages:,.1f}"],
        f"{selected_option.split(' ')[1]} Benchmark": [f"{b_item_ctr:.2%}", f"{b_bnr_ctr:.2%}", f"{b_pages:,.1f}"],
        "Variance vs Benchmark": [f"{c_item_ctr - b_item_ctr:+.2%} pts", f"{c_bnr_ctr - b_bnr_ctr:+.2%} pts", f"{c_pages - b_pages:+.1f} pages"]
    })
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        scorecard_df.to_excel(writer, sheet_name='Executive Scorecard', index=False)
    output.seek(0)
    
    dl_placeholder.download_button(
        label="⬇️ Download Benchmark Scorecard (.xlsx)",
        data=output,
        file_name=f"Benchmark_Scorecard.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.write("---")
    st.subheader(f"🎯 The Executive Scorecard vs Industry Average")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric(label="Client Avg. Item CTR", value=f"{c_item_ctr:.2%}", delta=f"{c_item_ctr - b_item_ctr:+.2%} pts vs Benchmark")
    sc2.metric(label="Client Marketing Banner CTR", value=f"{c_bnr_ctr:.2%}", delta=f"{c_bnr_ctr - b_bnr_ctr:+.2%} pts vs Benchmark")
    sc3.metric(label="Avg. Flyer Length (Pages)", value=f"{c_pages:,.1f}", delta=f"{c_pages - b_pages:+.1f} Pages vs Benchmark")

# ==============================================================================
# 🗺️ NAVIGATION & MAIN APP CONTROL
# ==============================================================================
st.sidebar.markdown("<h2 style='color:#002551;'>🚀 Control Panel</h2>", unsafe_allow_html=True)
pipeline_mode = st.sidebar.radio(
    "Select Strategy Module:", 
    [
        "📁 Single Campaign Breakdown", 
        "📊 Head-to-Head Comparison", 
        "🏆 Industry Benchmarks (DNU - IN DEV)",
        "🧰 Taylor's Workspace"
    ]
)

if "Single Campaign" in pipeline_mode: 
    render_single_campaign_matrix()
elif "Head-to-Head" in pipeline_mode: 
    render_head_to_head_variance()
elif "Industry Benchmarks" in pipeline_mode: 
    render_benchmark_scorecard()
elif "Taylor's Workspace" in pipeline_mode:
    render_taylors_workspace()
