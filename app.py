import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import re
import os
import gc

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
# ⚡ GLOBAL HIGH-CAPACITY FILE PARSER (Handles files up to 1 GB)
# ==============================================================================
@st.cache_data(show_spinner="⚡ Parsing dataset (Memory Optimized for Large Files)...")
def parse_large_file(file_bytes, file_name):
    """
    Streams and parses CSV/Excel files up to 1 GB in memory-efficient chunks.
    Downcasts numeric types and filters target columns to keep RAM under 100 MB.
    """
    if not file_bytes:
        return pd.DataFrame()
        
    buffer = io.BytesIO(file_bytes)
    
    target_columns = [
        'SKU', 'Clean_Name', 'Merchandise Name', 'Name', 'Item Name',
        'Total Item Views', 'Item Views', 'Views',
        'Total Item Clicks', 'Item Clicks', 'Clicks',
        'Total Transfer to Merchant (TTMs)', 'Item TTMs', 'Total TTMs', 'TTMs',
        'Page Position', 'Page', 'Display Type', 'Category', 'Retailer Category', 'Department'
    ]

    try:
        if file_name.lower().endswith('.csv'):
            preview = pd.read_csv(buffer, header=None, nrows=15)
            header_row = 0
            for idx, row in preview.iterrows():
                row_vals = [str(val).strip() for val in row.values if pd.notna(val)]
                if any(k in row_vals for k in ['Weekly Date', 'Daily Date', 'Flyer Run Name', 'Page Position', 'SKU', 'Name', 'Merchandise Name']):
                    header_row = idx
                    break

            buffer.seek(0)
            cols_in_file = pd.read_csv(buffer, header=header_row, nrows=1).columns.astype(str).str.strip()
            use_cols = [c for c in cols_in_file if c in target_columns or any(t in c.lower() for t in ['views', 'clicks', 'ttm', 'sku', 'name', 'page', 'category'])]

            buffer.seek(0)
            chunks = []
            chunk_iter = pd.read_csv(
                buffer,
                header=header_row,
                usecols=use_cols if use_cols else None,
                chunksize=50000,
                engine='c',
                low_memory=False
            )
            
            for chunk in chunk_iter:
                chunk.columns = chunk.columns.astype(str).str.strip()
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()

        else:
            df = pd.read_excel(buffer)
            df.columns = df.columns.astype(str).str.strip()

        # Standardize metric names
        rename_map = {
            'Total Item Views': 'Views', 'Item Views': 'Views',
            'Total Item Clicks': 'Clicks', 'Item Clicks': 'Clicks',
            'Total Transfer to Merchant (TTMs)': 'TTMs', 'Item TTMs': 'TTMs', 'Total TTMs': 'TTMs'
        }
        df.rename(columns=rename_map, inplace=True)

        for col in ['Views', 'Clicks', 'TTMs']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')
            else:
                df[col] = 0

        gc.collect()
        return df

    except Exception as e:
        st.error(f"⚠️ Error parsing file `{file_name}`: {str(e)}")
        return pd.DataFrame()

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
    
    is_creative = df['Display_Type'].isin(['BANNER', 'LINK']) | df['Name'].str.contains('BANNER', case=False, na=False)
    
    df_prod = df[~is_creative].copy()
    df_creative = df[is_creative].copy()
    
    return df_prod, df_creative, global_totals

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
# 🗂️ MODULE 1: CAMPAIGN PERFORMANCE BREAKDOWN (STREAMING & FAST)
# ==============================================================================
def render_single_campaign_matrix():
    import pandas as pd
    import numpy as np
    import io
    import re
    import plotly.express as px
    import streamlit as st

    def normalize_sale_story(val):
        if pd.isna(val) or str(val).strip() == "" or str(val).strip().lower() in ["nan", "none", "uncategorized / standard", "uncategorized"]:
            return "no badge"
        s = str(val).strip()
        m_pct_de_rabais = re.match(r'^(\d+(?:\.\d+)?)\s*%\s*de\s*rabais', s, flags=re.IGNORECASE)
        if m_pct_de_rabais: return f"SAVE {m_pct_de_rabais.group(1)}%"
        m_dollar_de_rabais = re.match(r'^(\d+(?:\.\d+)?)\s*\$\s*de\s*rabais', s, flags=re.IGNORECASE)
        if m_dollar_de_rabais: return f"SAVE ${m_dollar_de_rabais.group(1)}"
        m_rabais_pct = re.match(r'^rabais\s+de\s+(\d+(?:\.\d+)?)\s*%', s, flags=re.IGNORECASE)
        if m_rabais_pct: return f"SAVE {m_rabais_pct.group(1)}%"
        m_rabais_dollar = re.match(r'^rabais\s+de\s+(\d+(?:\.\d+)?)\s*\$', s, flags=re.IGNORECASE)
        if m_rabais_dollar: return f"SAVE ${m_rabais_dollar.group(1)}"
        if s.lower() in ["en solde", "on sale", "solde"]: return "ON SALE"
        s_clean = re.sub(r'^(ÉCONOMISEZ|ÉCONOMISER|ECONOMISEZ|ÉCONOMISE|RABAIS DE|RABAIS)\s*', 'SAVE ', s, flags=re.IGNORECASE).strip()
        s_clean = re.sub(r'^(LIQUIDATION)\s*', 'CLEARANCE ', s_clean, flags=re.IGNORECASE).strip()
        s_clean = re.sub(r'^(AUBAINE)\s*', 'HOT DEAL ', s_clean, flags=re.IGNORECASE).strip()
        s_clean = re.sub(r'SAVE\s+(\d+(?:\.\d+)?)\s*\$', r'SAVE $\1', s_clean, flags=re.IGNORECASE)
        s_clean = re.sub(r'SAVE\s+(\d+(?:\.\d+)?)\s*%', r'SAVE \1%', s_clean, flags=re.IGNORECASE)
        if s_clean.upper() in ["SAVE", "ÉCONOMISEZ"]: return "SAVE"
        return s_clean

    def clean_and_group_creative_assets(df_creative):
        if df_creative.empty: return df_creative
        df_cr = df_creative.copy()

        def scrub_creative_name(val):
            if pd.isna(val): return "Unassigned Asset"
            s = str(val).strip()
            s_clean = re.sub(r'-\d+$', '', s)
            s_clean = re.sub(r'_(EN|FR)_', '_', s_clean, flags=re.IGNORECASE)
            s_clean = re.sub(r'_(EN|FR)$', '', s_clean, flags=re.IGNORECASE)
            s_clean = re.sub(r'^(EN|FR)_', '', s_clean, flags=re.IGNORECASE)
            s_clean = re.sub(r'_{2,}', '_', s_clean).strip('_')
            return s_clean

        df_cr['Clean_Name'] = df_cr['Name'].apply(scrub_creative_name)
        if 'Page' not in df_cr.columns: df_cr['Page'] = 1
        if 'TTMs' not in df_cr.columns: df_cr['TTMs'] = 0

        cr_grouped = df_cr.groupby(['Clean_Name', 'Page'], observed=False).agg(
            Views=('Views', 'sum'),
            Clicks=('Clicks', 'sum'),
            TTMs=('TTMs', 'sum')
        ).reset_index()

        cr_grouped.rename(columns={'Clean_Name': 'Name'}, inplace=True)
        cr_grouped['Asset TTMR %'] = np.where(cr_grouped['Views'] > 0, cr_grouped['TTMs'] / cr_grouped['Views'], 0.0)
        cr_grouped['Asset CTR %'] = np.where(cr_grouped['Views'] > 0, cr_grouped['Clicks'] / cr_grouped['Views'], 0.0)
        return cr_grouped.sort_values(by=['TTMs', 'Clicks'], ascending=False)

    st.markdown("<div class='main-header'>Campaign Performance Breakdown</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload raw exports directly to analyze campaign performance across any selected flight package or timeframe.</div>", unsafe_allow_html=True)
    
    dl_placeholder = st.empty()
    
    col1, col2 = st.columns(2)
    with col1: merch_file = st.file_uploader("📁 Upload Merchandise File (.xlsx/.csv)", type=["xlsx", "csv"], key="m1_merch")
    with col2: scroll_file = st.file_uploader("📉 Upload Scroll Depth File (.xlsx/.csv)", type=["xlsx", "csv"], key="m1_scroll")
        
    if not merch_file and not scroll_file:
        st.info("⚠️ **Waiting for data:** Please upload a Merchandise file, a Scroll Depth file, or both to begin analysis.")
        return

    pivot_top = cat_l1_agg = cat_l2_agg = cat_l3_agg = brand_agg = cr_agg = p_agg = d_agg = s_agg_sorted = pd.DataFrame()
    df_sc_table = pd.DataFrame()
    weekly_scroll = pd.DataFrame()
    qbr_insights = None
    src_l1 = src_l2 = src_l3 = "N/A"
    col_sale_story = None

    # ⚙️ FAST MEMORY-OPTIMIZED FILE PARSER
    if merch_file:
        df_clean = parse_large_file(merch_file.getvalue(), merch_file.name)
        if not df_clean.empty:
            # Split Products vs Creative Banners
            item_col = 'Clean_Name' if 'Clean_Name' in df_clean.columns else ('Merchandise Name' if 'Merchandise Name' in df_clean.columns else 'Name')
            if item_col not in df_clean.columns:
                df_clean['Clean_Name'] = df_clean.iloc[:, 0].astype(str)
                item_col = 'Clean_Name'

            if 'SKU' in df_clean.columns:
                has_sku = df_clean['SKU'].notna() & (df_clean['SKU'].astype(str).str.strip() != '') & (df_clean['SKU'].astype(str).str.strip().str.lower() != 'nan')
                df_prod = df_clean[has_sku].copy()
                df_creative = df_clean[~has_sku].copy()
            elif 'Display Type' in df_clean.columns:
                is_link = df_clean['Display Type'].astype(str).str.contains('Banner|Header|Creative|Hero|Link', case=False, na=False)
                df_prod = df_clean[~is_link].copy()
                df_creative = df_clean[is_link].copy()
            else:
                df_prod = df_clean.copy()
                df_creative = pd.DataFrame()

            if 'Brand' not in df_prod.columns: df_prod['Brand'] = 'UNKNOWN'
            if 'Page' not in df_prod.columns and 'Page Position' in df_prod.columns: df_prod['Page'] = df_prod['Page Position']
            if 'Page' not in df_prod.columns: df_prod['Page'] = 1

            global_totals = {
                'views': df_clean['Views'].sum(),
                'clicks': df_clean['Clicks'].sum(),
                'clips': df_clean['Clips'].sum() if 'Clips' in df_clean.columns else 0,
                'ttms': df_clean['TTMs'].sum()
            }

            # Top items pivot
            grp_top = [item_col]
            if 'SKU' in df_prod.columns: grp_top.insert(0, 'SKU')
            pivot_top = df_prod.groupby(grp_top).agg({
                'Page': 'first', 'Views': 'sum', 'Clicks': 'sum', 'TTMs': 'sum'
            }).reset_index()
            pivot_top.rename(columns={item_col: 'Name'}, inplace=True)
            if 'SKU' not in pivot_top.columns: pivot_top['SKU'] = 'N/A'
            pivot_top['Clips'] = 0
            pivot_top['Item CTR'] = np.where(pivot_top['Views'] > 0, pivot_top['Clicks'] / pivot_top['Views'], 0.0)

            # Brand momentum
            brand_agg = df_prod.groupby('Brand').agg(
                Unique_Items=(item_col, 'nunique'), Views=('Views','sum'), Clicks=('Clicks','sum'), TTMs=('TTMs','sum')
            ).reset_index()
            brand_agg['Clips'] = 0
            brand_agg['Click Share %'] = brand_agg['Clicks'] / global_totals['clicks'] if global_totals['clicks'] > 0 else 0
            brand_agg['List Share %'] = 0
            brand_agg['TTM Share %'] = brand_agg['TTMs'] / global_totals['ttms'] if global_totals['ttms'] > 0 else 0

            if not df_creative.empty:
                df_creative.rename(columns={item_col: 'Name'}, inplace=True)
                cr_agg = clean_and_group_creative_assets(df_creative)

    if scroll_file:
        try:
            df_sc_raw, weekly_scroll, qbr_insights = process_scroll_file(scroll_file)
            df_sc_table = df_sc_raw.copy().rename(columns={'Milestone': 'Scroll Depth', 'Retention': '% of Users Read'})
        except Exception as e:
            st.warning(f"Could not process the scroll file: {str(e)}")

    # RENDER DASHBOARD METRICS
    if merch_file and not df_clean.empty:
        st.write("---")
        v_tot, cl_tot, cp_tot, t_tot = global_totals['views'], global_totals['clicks'], global_totals['clips'], global_totals['ttms']
        ctr_global_display = f"{cl_tot/v_tot:.2%}" if v_tot > 0 else "0.00%"
        
        item_v_tot = df_prod['Views'].sum() if not df_prod.empty else 0
        item_cl_tot = df_prod['Clicks'].sum() if not df_prod.empty else 0
        item_t_tot = df_prod['TTMs'].sum() if not df_prod.empty else 0
        item_ctr_display = f"{item_cl_tot/item_v_tot:.2%}" if item_v_tot > 0 else "0.00%"
        
        st.markdown("<h4 style='color:#002551; margin-top:20px;'>🌐 Overall Campaign Totals (Includes Banners)</h4>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f"<div class='metric-card'><div class='metric-val'>{v_tot:,.0f}</div><div class='metric-lbl'>TOTAL VIEWS</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><div class='metric-val'>{cl_tot:,.0f}</div><div class='metric-lbl'>TOTAL CLICKS</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><div class='metric-val'>{cp_tot:,.0f}</div><div class='metric-lbl'>ADD TO LISTS</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><div class='metric-val'>{t_tot:,.0f}</div><div class='metric-lbl'>TOTAL TTMS</div></div>", unsafe_allow_html=True)
        c5.markdown(f"<div class='metric-card'><div class='metric-val'>{ctr_global_display}</div><div class='metric-lbl'>GLOBAL CTR</div></div>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='color:#002551; margin-top:20px;'>🛒 Item-Specific Performance (Products Only)</h4>", unsafe_allow_html=True)
        i1, i2, i3, i4, i5 = st.columns(5)
        i1.markdown(f"<div class='metric-card'><div class='metric-val'>{item_v_tot:,.0f}</div><div class='metric-lbl'>TOTAL ITEM VIEWS</div></div>", unsafe_allow_html=True)
        i2.markdown(f"<div class='metric-card'><div class='metric-val'>{item_cl_tot:,.0f}</div><div class='metric-lbl'>ITEM CLICKS</div></div>", unsafe_allow_html=True)
        i3.markdown(f"<div class='metric-card'><div class='metric-val'>0</div><div class='metric-lbl'>ITEM ADD TO LISTS</div></div>", unsafe_allow_html=True)
        i4.markdown(f"<div class='metric-card'><div class='metric-val'>{item_t_tot:,.0f}</div><div class='metric-lbl'>ITEM TTMS</div></div>", unsafe_allow_html=True)
        i5.markdown(f"<div class='metric-card'><div class='metric-val'>{item_ctr_display}</div><div class='metric-lbl'>ITEM CTR</div></div>", unsafe_allow_html=True)
        
        st.write("---")
        st.subheader("🏆 Top 10 Items by Total Clicks (Volume)")
        st.dataframe(pivot_top[['SKU', 'Name', 'Page', 'Views', 'Clicks', 'TTMs', 'Item CTR']].sort_values(by='Clicks', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)

        st.write("---")
        st.subheader("🎯 Top 10 Items by Item CTR (Efficiency)")
        st.dataframe(pivot_top[['SKU', 'Name', 'Page', 'Views', 'Clicks', 'TTMs', 'Item CTR']].sort_values(by='Item CTR', ascending=False).head(10).style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'TTMs': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)

        if not cr_agg.empty:
            st.write("---")
            st.subheader("🖼️ Consolidated Marketing Banners (Ranked by TTMR %)")
            st.dataframe(
                cr_agg[['Name', 'Page', 'Views', 'Clicks', 'TTMs', 'Asset TTMR %', 'Asset CTR %']]
                .style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'TTMs': '{:,.0f}', 'Asset TTMR %': '{:.2%}', 'Asset CTR %': '{:.2%}'}),
                use_container_width=True, hide_index=True
            )
# ==============================================================================
# 🗂️ MODULE 2: HEAD-TO-HEAD COMPARISON (PERSISTENT & FAST)
# ==============================================================================
def render_head_to_head_variance():
    st.write("---")
    st.header("⚖️ Head-to-Head Campaign Comparison")
    st.markdown("Upload your Base (Historical) and New (Current) campaign files to generate period-over-period variance and side-by-side performance tables.")

    st.markdown("### 🛒 Merchandise Metrics")
    col1, col2 = st.columns(2)
    with col1:
        base_merch_file = st.file_uploader("📤 Upload BASE Merchandise Metrics", type=['csv', 'xlsx'], key="base_merch")
    with col2:
        new_merch_file = st.file_uploader("📤 Upload NEW Merchandise Metrics", type=['csv', 'xlsx'], key="new_merch")

    st.markdown("### 📊 Optional: Funnel Metrics")
    st.info("Upload Base and New Funnel Metrics to unlock Macro Performance comparison. Runs independently without Merchandise files.")
    
    col3, col4 = st.columns(2)
    with col3:
        base_funnel_file = st.file_uploader("📤 Upload BASE Funnel Metrics", type=['csv', 'xlsx'], key="base_funnel")
    with col4:
        new_funnel_file = st.file_uploader("📤 Upload NEW Funnel Metrics", type=['csv', 'xlsx'], key="new_funnel")

    if "run_h2h" not in st.session_state:
        st.session_state.run_h2h = False

    if st.button("🚀 Run Head-to-Head Analysis", type="primary"):
        st.session_state.run_h2h = True

    if st.session_state.run_h2h:
        export_sheets = {}

        df_base = parse_large_file(base_merch_file.getvalue(), base_merch_file.name) if base_merch_file else pd.DataFrame()
        df_new = parse_large_file(new_merch_file.getvalue(), new_merch_file.name) if new_merch_file else pd.DataFrame()
        
        f_base = parse_large_file(base_funnel_file.getvalue(), base_funnel_file.name) if base_funnel_file else pd.DataFrame()
        f_new = parse_large_file(new_funnel_file.getvalue(), new_funnel_file.name) if new_funnel_file else pd.DataFrame()

        # 1. FUNNEL ANALYSIS
        if not f_base.empty and not f_new.empty:
            st.success("✅ Both Funnel files loaded!")

            def extract_funnel_metrics(df):
                def safe_sum(col_names):
                    for col in col_names:
                        if col in df.columns:
                            return pd.to_numeric(df[col], errors='coerce').fillna(0).sum()
                    return 0

                impressions = safe_sum(['Total Flyer Impressions', 'Impressions'])
                opens = safe_sum(['Total Opens', 'Flyer Opens'])
                uevs = safe_sum(['Total Unique Engaged Visits (UEVs)', 'Unique Engagements'])
                clicks = safe_sum(['Total Item Clicks', 'Total Flyer Clicks', 'Item Clicks'])
                ttms = safe_sum(['Total Transfer to Merchant (TTMs)', 'Total Transfer to Site', 'TTMs'])
                adds = safe_sum(['Total Clippings', 'Total Shopping List Adds', 'Clippings'])
                tot_time_sec = safe_sum(['Total Time On Flyer (Sec)', 'Total Time Spent (Sec)'])
                tot_sessions = safe_sum(['Total Flyer Sessions', 'Flyer Sessions'])

                avg_time_sec = (tot_time_sec / tot_sessions) if tot_sessions > 0 else 0
                open_rate = (opens / impressions) if impressions > 0 else 0
                eng_rate = (uevs / opens) if opens > 0 else 0
                ctor = (clicks / opens) if opens > 0 else 0
                intent_rate = (adds / uevs) if uevs > 0 else 0

                formatted_time = f"{(avg_time_sec / 60):.2f} min" if avg_time_sec >= 60 else f"{avg_time_sec:.1f} sec"

                return {
                    "Impressions": impressions, "Flyer Opens": opens, "Open Rate %": open_rate,
                    "Unique Engagements": uevs, "Engagement Rate %": eng_rate, "Total Flyer Clicks": clicks,
                    "CTOR %": ctor, "Transfer to Site": ttms, "Shopping List Adds": adds,
                    "Intent Rate %": intent_rate, "Raw Avg Time Sec": avg_time_sec, "Formatted Time": formatted_time
                }

            b_m = extract_funnel_metrics(f_base)
            n_m = extract_funnel_metrics(f_new)

            def calc_var(new_v, base_v):
                return (new_v - base_v) / base_v if base_v > 0 else 0.0

            st.subheader("🚀 Top-of-Funnel Macro Performance Comparison")
            funnel_data = {
                "Metric": ["Historical (Base)", "Current (New)", "Variance %"],
                "Impressions": [f"{b_m['Impressions']:,.0f}", f"{n_m['Impressions']:,.0f}", f"{calc_var(n_m['Impressions'], b_m['Impressions']):+.2%}"],
                "Flyer Opens": [f"{b_m['Flyer Opens']:,.0f}", f"{n_m['Flyer Opens']:,.0f}", f"{calc_var(n_m['Flyer Opens'], b_m['Flyer Opens']):+.2%}"],
                "Open Rate %": [f"{b_m['Open Rate %']:.2%}", f"{n_m['Open Rate %']:.2%}", f"{calc_var(n_m['Open Rate %'], b_m['Open Rate %']):+.2%}"],
                "Unique Engagements": [f"{b_m['Unique Engagements']:,.0f}", f"{n_m['Unique Engagements']:,.0f}", f"{calc_var(n_m['Unique Engagements'], b_m['Unique Engagements']):+.2%}"],
                "Engagement Rate %": [f"{b_m['Engagement Rate %']:.2%}", f"{n_m['Engagement Rate %']:.2%}", f"{calc_var(n_m['Engagement Rate %'], b_m['Engagement Rate %']):+.2%}"],
                "Total Flyer Clicks": [f"{b_m['Total Flyer Clicks']:,.0f}", f"{n_m['Total Flyer Clicks']:,.0f}", f"{calc_var(n_m['Total Flyer Clicks'], b_m['Total Flyer Clicks']):+.2%}"],
                "CTOR %": [f"{b_m['CTOR %']:.2%}", f"{n_m['CTOR %']:.2%}", f"{calc_var(n_m['CTOR %'], b_m['CTOR %']):+.2%}"],
                "Transfer to Site": [f"{b_m['Transfer to Site']:,.0f}", f"{n_m['Transfer to Site']:,.0f}", f"{calc_var(n_m['Transfer to Site'], b_m['Transfer to Site']):+.2%}"],
                "Shopping List Adds": [f"{b_m['Shopping List Adds']:,.0f}", f"{n_m['Shopping List Adds']:,.0f}", f"{calc_var(n_m['Shopping List Adds'], b_m['Shopping List Adds']):+.2%}"],
                "Intent Rate %": [f"{b_m['Intent Rate %']:.2%}", f"{n_m['Intent Rate %']:.2%}", f"{calc_var(n_m['Intent Rate %'], b_m['Intent Rate %']):+.2%}"],
                "Avg Time Spent": [b_m['Formatted Time'], n_m['Formatted Time'], f"{calc_var(n_m['Raw Avg Time Sec'], b_m['Raw Avg Time Sec']):+.2%}"]
            }
            df_funnel_summary = pd.DataFrame(funnel_data)
            export_sheets["Funnel_Macro"] = df_funnel_summary
            st.dataframe(df_funnel_summary, use_container_width=True, hide_index=True)

        # 2. MERCHANDISE ANALYSIS
        if not df_base.empty and not df_new.empty:
            st.success("✅ Both Merchandise files loaded!")

            item_col = 'Clean_Name' if 'Clean_Name' in df_base.columns else ('Merchandise Name' if 'Merchandise Name' in df_base.columns else 'Name')

            def get_group_cols(df, main_c):
                cols = [main_c] if main_c in df.columns else []
                if 'Flyer Run Name' in df.columns: cols.append('Flyer Run Name')
                if 'Page Position' in df.columns: cols.append('Page Position')
                return cols

            if 'Views' in df_base.columns and 'Clicks' in df_base.columns:
                st.write("---")
                st.subheader("📈 Macro Item & Marketing Link Performance Comparison")

                def split_items_and_links(df):
                    if df.empty: return pd.DataFrame(), pd.DataFrame()
                    if 'SKU' in df.columns:
                        has_sku = df['SKU'].notna() & (df['SKU'].astype(str).str.strip() != '') & (df['SKU'].astype(str).str.strip().str.lower() != 'nan')
                        return df[has_sku].copy(), df[~has_sku].copy()
                    elif 'Display Type' in df.columns:
                        is_link = df['Display Type'].astype(str).str.contains('Banner|Header|Creative|Hero|Link', case=False, na=False)
                        return df[~is_link].copy(), df[is_link].copy()
                    return df.copy(), pd.DataFrame()

                b_items, b_links = split_items_and_links(df_base)
                n_items, n_links = split_items_and_links(df_new)

                b_item_v = b_items['Views'].sum() if not b_items.empty else 0
                b_item_cl = b_items['Clicks'].sum() if not b_items.empty else 0
                b_item_ttm = b_items['TTMs'].sum() if not b_items.empty else 0
                
                b_link_v = b_links['Views'].sum() if not b_links.empty else 0
                b_link_cl = b_links['Clicks'].sum() if not b_links.empty else 0
                b_link_ttm = b_links['TTMs'].sum() if not b_links.empty else 0

                n_item_v = n_items['Views'].sum() if not n_items.empty else 0
                n_item_cl = n_items['Clicks'].sum() if not n_items.empty else 0
                n_item_ttm = n_items['TTMs'].sum() if not n_items.empty else 0

                n_link_v = n_links['Views'].sum() if not n_links.empty else 0
                n_link_cl = n_links['Clicks'].sum() if not n_links.empty else 0
                n_link_ttm = n_links['TTMs'].sum() if not n_links.empty else 0

                b_tot_v, b_tot_cl, b_tot_ttm = b_item_v + b_link_v, b_item_cl + b_link_cl, b_item_ttm + b_link_ttm
                n_tot_v, n_tot_cl, n_tot_ttm = n_item_v + n_link_v, n_item_cl + n_link_cl, n_item_ttm + n_link_ttm

                b_item_ctr = (b_item_cl / b_item_v) if b_item_v > 0 else 0.0
                n_item_ctr = (n_item_cl / n_item_v) if n_item_v > 0 else 0.0

                b_link_ttmr = (b_link_ttm / b_link_v) if b_link_v > 0 else 0.0
                n_link_ttmr = (n_link_ttm / n_link_v) if n_link_v > 0 else 0.0

                b_tot_ctr = (b_tot_cl / b_tot_v) if b_tot_v > 0 else 0.0
                n_tot_ctr = (n_tot_cl / n_tot_v) if n_tot_v > 0 else 0.0

                def calc_var(new_v, base_v):
                    return (new_v - base_v) / base_v if base_v > 0 else 0.0

                summary_data = {
                    "Metric": ["Historical (Base)", "Current (New)", "Variance %"],
                    "Product Views": [f"{b_item_v:,.0f}", f"{n_item_v:,.0f}", f"{calc_var(n_item_v, b_item_v):+.2%}"],
                    "Product Clicks": [f"{b_item_cl:,.0f}", f"{n_item_cl:,.0f}", f"{calc_var(n_item_cl, b_item_cl):+.2%}"],
                    "Product CTR %": [f"{b_item_ctr:.2%}", f"{n_item_ctr:.2%}", f"{calc_var(n_item_ctr, b_item_ctr):+.2%}"],
                    "Product TTMs": [f"{b_item_ttm:,.0f}", f"{n_item_ttm:,.0f}", f"{calc_var(n_item_ttm, b_item_ttm):+.2%}"],
                    "Link Views": [f"{b_link_v:,.0f}", f"{n_link_v:,.0f}", f"{calc_var(n_link_v, b_link_v):+.2%}"],
                    "Link Clicks": [f"{b_link_cl:,.0f}", f"{n_link_cl:,.0f}", f"{calc_var(n_link_cl, b_link_cl):+.2%}"],
                    "Link TTMR %": [f"{b_link_ttmr:.2%}", f"{n_link_ttmr:.2%}", f"{calc_var(n_link_ttmr, b_link_ttmr):+.2%}"],
                    "Total Views": [f"{b_tot_v:,.0f}", f"{n_tot_v:,.0f}", f"{calc_var(n_tot_v, b_tot_v):+.2%}"],
                    "Total Clicks": [f"{b_tot_cl:,.0f}", f"{n_tot_cl:,.0f}", f"{calc_var(n_tot_cl, b_tot_cl):+.2%}"],
                    "Global CTR %": [f"{b_tot_ctr:.2%}", f"{n_tot_ctr:.2%}", f"{calc_var(n_tot_ctr, b_tot_ctr):+.2%}"],
                    "Total TTMs": [f"{b_tot_ttm:,.0f}", f"{n_tot_ttm:,.0f}", f"{calc_var(n_tot_ttm, b_tot_ttm):+.2%}"]
                }

                df_summary = pd.DataFrame(summary_data)
                export_sheets["Merch_Macro"] = df_summary
                st.dataframe(df_summary, use_container_width=True, hide_index=True)

            if 'Page Position' in df_base.columns and 'Page Position' in df_new.columns:
                st.write("---")
                st.subheader("📖 Page-by-Page Engagement Analysis")
                
                df_base['Page Position'] = df_base['Page Position'].astype(str).str.replace(".0", "", regex=False).str.strip()
                df_new['Page Position'] = df_new['Page Position'].astype(str).str.replace(".0", "", regex=False).str.strip()

                base_p_agg = df_base.groupby('Page Position').agg({'Views': 'sum', 'Clicks': 'sum'}).reset_index()
                base_p_agg['Base CTR %'] = np.where(base_p_agg['Views'] > 0, base_p_agg['Clicks'] / base_p_agg['Views'], 0)
                base_p_agg.rename(columns={'Views': 'Base Views', 'Clicks': 'Base Clicks'}, inplace=True)

                new_p_agg = df_new.groupby('Page Position').agg({'Views': 'sum', 'Clicks': 'sum'}).reset_index()
                new_p_agg['New CTR %'] = np.where(new_p_agg['Views'] > 0, new_p_agg['Clicks'] / new_p_agg['Views'], 0)
                new_p_agg.rename(columns={'Views': 'New Views', 'Clicks': 'New Clicks'}, inplace=True)

                merged_page = pd.merge(base_p_agg, new_p_agg, on='Page Position', how='outer').fillna(0)
                merged_page['Page_Num'] = pd.to_numeric(merged_page['Page Position'], errors='coerce')
                merged_page = merged_page.sort_values(by='Page_Num').drop(columns=['Page_Num'])

                export_sheets["Page_Engagement"] = merged_page
                st.dataframe(
                    merged_page.style.format({
                        'Base Views': '{:,.0f}', 'Base Clicks': '{:,.0f}', 'Base CTR %': '{:.2%}',
                        'New Views': '{:,.0f}', 'New Clicks': '{:,.0f}', 'New CTR %': '{:.2%}'
                    }),
                    use_container_width=True, hide_index=True
                )

            def filter_assets(df):
                if 'SKU' in df.columns:
                    no_sku = df['SKU'].isna() | (df['SKU'].astype(str).str.strip() == '') | (df['SKU'].astype(str).str.strip().str.lower() == 'nan')
                    return df[no_sku].copy()
                if 'Display Type' in df.columns:
                    return df[df['Display Type'].astype(str).str.upper() == 'LINK'].copy()
                return pd.DataFrame()
                
            df_base_assets = filter_assets(df_base)
            df_new_assets = filter_assets(df_new)

            if item_col in df_base_assets.columns and item_col in df_new_assets.columns:
                if len(df_base_assets) > 0 or len(df_new_assets) > 0:
                    base_grp = get_group_cols(df_base_assets, item_col)
                    new_grp = get_group_cols(df_new_assets, item_col)

                    base_a_agg = df_base_assets.groupby(base_grp).agg({'Views': 'sum', 'Clicks': 'sum', 'TTMs': 'sum'}).reset_index()
                    base_a_agg['TTMR %'] = np.where(base_a_agg['Views'] > 0, base_a_agg['TTMs'] / base_a_agg['Views'], 0)
                    
                    new_a_agg = df_new_assets.groupby(new_grp).agg({'Views': 'sum', 'Clicks': 'sum', 'TTMs': 'sum'}).reset_index()
                    new_a_agg['TTMR %'] = np.where(new_a_agg['Views'] > 0, new_a_agg['TTMs'] / new_a_agg['Views'], 0)

                    base_pool = base_a_agg[base_a_agg['Views'] >= 50] if len(base_a_agg[base_a_agg['Views'] >= 50]) > 0 else base_a_agg
                    new_pool = new_a_agg[new_a_agg['Views'] >= 50] if len(new_a_agg[new_a_agg['Views'] >= 50]) > 0 else new_a_agg

                    st.write("---")
                    st.subheader("🖼️ Top-10 Marketing Assets by TTMR %")
                    
                    base_top_ttmr = base_pool.sort_values(by=['TTMR %', 'TTMs'], ascending=[False, False]).head(10)[base_grp + ['TTMs', 'TTMR %']]
                    base_top_ttmr.rename(columns={item_col: 'Asset Name'}, inplace=True)
                    
                    new_top_ttmr = new_pool.sort_values(by=['TTMR %', 'TTMs'], ascending=[False, False]).head(10)[new_grp + ['TTMs', 'TTMR %']]
                    new_top_ttmr.rename(columns={item_col: 'Asset Name'}, inplace=True)

                    export_sheets["Assets_Top_TTMR_Base"] = base_top_ttmr
                    export_sheets["Assets_Top_TTMR_New"] = new_top_ttmr

                    c7, c8 = st.columns(2)
                    with c7:
                        st.markdown("**Historical Top TTMR % (Base)**")
                        st.dataframe(base_top_ttmr.style.format({'TTMs': '{:,.0f}', 'TTMR %': '{:.2%}'}), use_container_width=True, hide_index=True)
                    with c8:
                        st.markdown("**Current Top TTMR % (New)**")
                        st.dataframe(new_top_ttmr.style.format({'TTMs': '{:,.0f}', 'TTMR %': '{:.2%}'}), use_container_width=True, hide_index=True)

                    st.write("---")
                    st.subheader("📢 Top-10 Marketing Assets by Total TTMs")
                    
                    base_top_ttms = base_a_agg.sort_values(by='TTMs', ascending=False).head(10)[base_grp + ['TTMs', 'TTMR %']]
                    base_top_ttms.rename(columns={item_col: 'Asset Name'}, inplace=True)
                    
                    new_top_ttms = new_a_agg.sort_values(by='TTMs', ascending=False).head(10)[new_grp + ['TTMs', 'TTMR %']]
                    new_top_ttms.rename(columns={item_col: 'Asset Name'}, inplace=True)

                    export_sheets["Assets_Top_TTMs_Base"] = base_top_ttms
                    export_sheets["Assets_Top_TTMs_New"] = new_top_ttms

                    c9, c10 = st.columns(2)
                    with c9:
                        st.markdown("**Historical Top TTM Volume (Base)**")
                        st.dataframe(base_top_ttms.style.format({'TTMs': '{:,.0f}', 'TTMR %': '{:.2%}'}), use_container_width=True, hide_index=True)
                    with c10:
                        st.markdown("**Current Top TTM Volume (New)**")
                        st.dataframe(new_top_ttms.style.format({'TTMs': '{:,.0f}', 'TTMR %': '{:.2%}'}), use_container_width=True, hide_index=True)

        if export_sheets:
            st.write("---")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df_sheet in export_sheets.items():
                    clean_sheet = re.sub(r'[\\/*?:\[\]]', '_', str(sheet_name))[:30]
                    df_sheet.to_excel(writer, sheet_name=clean_sheet, index=False)
            
            st.download_button(
                label="📥 Download Campaign Analysis (.xlsx)",
                data=output.getvalue(),
                file_name="Head_to_Head_Campaign_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        if not (base_merch_file and new_merch_file) and not (base_funnel_file and new_funnel_file):
            st.warning("⚠️ Please upload BOTH Base and New files for either Merchandise or Funnel metrics to run the comparison.")

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
# 🧰 MODULE 4: TAYLOR'S WORKSPACE (REGIONAL CTR ENGINE)
# ==============================================================================
@st.cache_data
def load_usps_reference(path):
    if path.endswith('.csv'):
        df = pd.read_csv(path, dtype=str, low_memory=False) 
    else:
        df = pd.read_excel(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()]
    
def render_taylors_workspace():
    st.markdown("<div class='main-header'>🧰 Taylor's Regional CTR Engine</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload your Merch Metrics and FSA Zone file(s) to instantly join and calculate regional performance. The USPS reference is loaded automatically from the server. No VLOOKUPs required.</div>", unsafe_allow_html=True)

    dl_placeholder = st.empty()

    col1, col2 = st.columns(2)
    with col1: merch_file = st.file_uploader("1️⃣ Upload Merchandise Metrics", type=["xlsx", "csv"])
    with col2: fsa_files = st.file_uploader("2️⃣ Upload FSA Zone Reports (Multiple Allowed)", type=["xlsx", "csv"], accept_multiple_files=True)

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

        df_usps = load_usps_reference(usps_path)

        def get_col_fuzzy_strict(df, keywords, exclude_cols=None):
            exclude_cols = exclude_cols or []
            for k in keywords:
                for col in df.columns:
                    if col not in exclude_cols and str(col).strip().lower() == k: return col
            for k in keywords:
                for col in df.columns:
                    if col not in exclude_cols and k in str(col).lower(): return col
            return "UNKNOWN"

        fsa_desc_col = get_col_fuzzy_strict(df_fsa, ['pricing zone name', 'description', 'flyer', 'campaign', 'name', 'zone'])
        if fsa_desc_col == "UNKNOWN": fsa_desc_col = df_fsa.columns[0]

        exclude_cols = [fsa_desc_col]
        id_col = get_col_fuzzy_strict(df_fsa, ['pricing zone id', 'id'])
        if id_col != "UNKNOWN": exclude_cols.append(id_col)

        fsa_zip_col = get_col_fuzzy_strict(df_fsa, ['fsa', 'zip', 'postal'], exclude_cols=exclude_cols)
        if fsa_zip_col == "UNKNOWN": fsa_zip_col = [c for c in df_fsa.columns if c not in exclude_cols][0]

        usps_zip_col = get_col_fuzzy_strict(df_usps, ['fsa', 'zip', 'postal'])
        if usps_zip_col == "UNKNOWN": usps_zip_col = df_usps.columns[0]

        usps_state_col = get_col_fuzzy_strict(df_usps, ['state', 'province', 'st', 'region', 'terr'], exclude_cols=[usps_zip_col])
        if usps_state_col == "UNKNOWN": usps_state_col = [c for c in df_usps.columns if c != usps_zip_col][0]

        def safe_pad_zip(z):
            z = str(z).strip().upper().replace(' ', '').replace('.0', '')
            if z == 'NAN' or z == 'NONE': return 'UNKNOWN'
            if z.isdigit() and len(z) < 5: return z.zfill(5)
            return z

        df_fsa[fsa_zip_col] = df_fsa[fsa_zip_col].apply(safe_pad_zip)
        df_usps[usps_zip_col] = df_usps[usps_zip_col].apply(safe_pad_zip)

        def aggressive_key_clean(s):
            cleaned = re.sub(r'[^A-Z0-9]', '', str(s).upper())
            if cleaned.startswith("ZONE") and len(cleaned) > 4: cleaned = cleaned.replace("ZONE", "")
            return cleaned

        df_prod['Flyer_Join_Key'] = df_prod['Flyer_Description'].apply(aggressive_key_clean)
        df_fsa['FSA_Join_Key'] = df_fsa[fsa_desc_col].apply(aggressive_key_clean)

        df_usps_unique = df_usps[[usps_zip_col, usps_state_col]].drop_duplicates(subset=[usps_zip_col])
        campaign_zips = df_fsa.merge(df_usps_unique, left_on=fsa_zip_col, right_on=usps_zip_col, how='inner')
        campaign_states = campaign_zips[['FSA_Join_Key', usps_state_col]].drop_duplicates()

        def assign_custom_region(state_code):
            st_clean = str(state_code).strip().upper()
            if st_clean in ['DE', 'MD', 'NJ', 'OH', 'PA', 'VA', 'DC', 'WV', 'IN', 'NC', 'DELAWARE', 'MARYLAND', 'NEW JERSEY', 'OHIO', 'PENNSYLVANIA', 'VIRGINIA', 'DISTRICT OF COLUMBIA', 'WEST VIRGINIA', 'INDIANA', 'NORTH CAROLINA']: return 'East'
            if st_clean in ['CA', 'AZ', 'CALIFORNIA', 'ARIZONA']: return 'West'
            if st_clean in ['ID', 'OR', 'WA', 'MT', 'IDAHO', 'OREGON', 'WASHINGTON', 'MONTANA']: return 'Northwest'
            if st_clean in ['LV', 'NV', 'NEVADA', 'LAS VEGAS']: return 'Nevada'
            return 'Other'

        campaign_states['Region'] = campaign_states[usps_state_col].apply(assign_custom_region)
        campaign_region_map = campaign_states[['FSA_Join_Key', 'Region']].drop_duplicates(subset=['FSA_Join_Key'], keep='first')
        df_prod = df_prod.merge(campaign_region_map, left_on='Flyer_Join_Key', right_on='FSA_Join_Key', how='left')

        unmatched_mask = df_prod['Region'].isna()
        if unmatched_mask.any():
            fallback_list = campaign_region_map.dropna(subset=['FSA_Join_Key', 'Region']).values.tolist()
            def smarter_match(m_key):
                m_str = str(m_key).strip().lower()
                if not m_str or m_str in ['nan', 'none']: return 'Other'
                m_nums = set([str(int(n)) for n in re.findall(r'\d+', m_str)])
                for f_key, reg in fallback_list:
                    f_nums = set([str(int(n)) for n in re.findall(r'\d+', str(f_key))])
                    if f_nums and f_nums.issubset(m_nums): return reg
                m_words = set(re.findall(r'\b\w+\b', m_str))
                for f_key, reg in fallback_list:
                    f_words = set(re.findall(r'\b\w+\b', str(f_key).lower()))
                    if f_words and f_words.issubset(m_words): return reg
                for f_key, reg in fallback_list:
                    f_clean = str(f_key).strip().lower()
                    if len(f_clean) >= 4 and f_clean in m_str: return reg
                return 'Other'
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

        def get_taylor_cat(name):
            text = f" {name} ".lower()

            if 'churu' in text: return 'Pet'
            if any(w in text for w in ['charcuterie', 'buddig', 'smithfield', 'columbus', 'foster farms']): return 'Deli'
            if any(w in text for w in ['jerky', 'beef stick', 'protein bar', 'snack bar', 'chocolate bar', 'rxbar', 'granola', 'cracker']): return 'Grocery'
            if any(w in text for w in ['salad', 'cucumbers', 'watermelons', 'papayas', 'peaches', 'nectarines', 'bananas', 'onions', 'lemons', 'limes', 'avocados', 'cherries', 'tomatoes', 'corn', 'grapes', 'mangos', 'strawberries', 'blueberries', 'raspberries']): return 'Produce' 
            if any(w in text for w in ['freezer pop', 'jimmy dean', 'skillet meal', 'popcorn chicken', 'nugget', 'breaded chicken', 'bowl']): return 'Frozen'
            if any(w in text for w in ['iced tea', 'coconut water', 'iced coffee', 'tropicana', 'juice']): return 'Beverages'
            if any(w in text for w in ['cream cheese', 'cottage cheese']): return 'Dairy' 
            if 'yasso' in text: return 'Ice Cream'

            if re.search(r'\b(wine|beer|spirit|spirits|vodka|whiskey|rum|gin|tequila|cooler|cider|ale|lager|liquor|alcohol)\b', text): return 'Alcohol'
            if 'bacon' in text: return 'Bacon'
            if any(w in text for w in ['butter', 'margarine', 'ghee']) and not any(w in text for w in ['peanut', 'almond']): return 'Butter'
            if any(w in text for w in ['ice cream', 'gelato', 'sorbet', 'popsicle', 'freezie']): return 'Ice Cream'
            if any(w in text for w in ['cheese', 'cheddar', 'mozzarella', 'brie', 'feta', 'parmesan', 'provolone', 'gouda']): return 'Cheese'
            if any(w in text for w in ['milk', 'yogurt', 'cream', 'oat', 'soy', 'dairy']): return 'Dairy'
            if 'egg' in text and not any(w in text for w in ['chocolate', 'easter', 'cadbury', 'leg']): return 'Eggs'
            if any(w in text for w in ['frozen', 'pizza', 'waffle']) and not any(w in text for w in ['bread', 'pie']): return 'Frozen'
            if any(w in text for w in ['salmon', 'shrimp', 'cod', 'tuna', 'fish', 'lobster', 'crab', 'scallop', 'seafood', 'oyster', 'tilapia']): return 'Seafood'
            if any(w in text for w in ['beef', 'chicken', 'pork', 'steak', 'ground', 'ribs', 'chops', 'veal', 'lamb', 'turkey', 'sausage', 'burger', 'crooked willow', 'poultry', 'meat', 'roast', 'breast', 'thigh']): return 'Fresh Meat'
            if any(w in text for w in ['deli', 'cold cut', 'salami', 'prosciutto', 'ham', 'hummus', 'roast beef']): return 'Deli'
            if any(w in text for w in ['bread', 'bun', 'croissant', 'muffin', 'bagel', 'cake', 'pie', 'pastry', 'tart', 'bakery']) and not any(w in text for w in ['oreo', 'cookie', 'frozen', 'bar']): return 'Bakery'
            if any(w in text for w in ['apple', 'banana', 'lettuce', 'tomato', 'potato', 'onion', 'fruit', 'vegetable', 'berries', 'grape', 'orange', 'carrot', 'broccoli', 'produce']): return 'Produce'
            if re.search(r'\b(juice|pop|soda|water|coffee|tea|coke|pepsi|sprite|beverage|drink)\b', text): return 'Beverages'
            if re.search(r'\b(paper towel|toilet paper|detergent|cleaner|foil|garbage bag|soap|shampoo|toothpaste|tissue|napkin|trash bag|home|cutlery)\b', text): return 'Home'
            if re.search(r'\b(cat|dog|pet|litter|kibble|purina|treat|churu)\b', text): return 'Pet'

            return 'Grocery'

        df_prod['cat_m'] = df_prod['Clean_Name'].apply(get_taylor_cat)

        override_filepath = "reclassified_products_2.xlsx"
        if os.path.exists(override_filepath):
            try:
                df_overrides = pd.read_excel(override_filepath)
                override_dict = dict(zip(
                    df_overrides['Name'].astype(str).str.lower().str.strip(), 
                    df_overrides['Reassigned Category'].astype(str).str.strip()
                ))
                
                def apply_taylors_override(row):
                    item_name = str(row['Clean_Name']).lower().strip()
                    if item_name in override_dict and pd.notna(override_dict[item_name]):
                        return override_dict[item_name]
                    return row['cat_m']
                    
                df_prod['cat_m'] = df_prod.apply(apply_taylors_override, axis=1)
                df_prod['L1_Category'] = df_prod['cat_m']
                df_prod['Category'] = df_prod['cat_m']
                
                if 'df_prod' in st.session_state:
                    st.session_state['df_prod'] = df_prod
                    
            except Exception as e:
                st.warning(f"⚠️ Found the override file, but couldn't read it: {e}")

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

        st.markdown("**3. Flyer Name Handshake Test**")
        st.write("First 5 Flyer Names in Merch File:", df_prod['Flyer_Join_Key'].head(5).tolist())
        st.write("First 5 Flyer Names in FSA File:", df_fsa['FSA_Join_Key'].head(5).tolist())

    st.success("✅ **Data Merged Successfully!** Blank SKUs filtered, categories assigned correctly, and regions matched.")

    cat_agg = df_prod.groupby('cat_m').agg({'Views': 'sum', 'Clicks': 'sum'}).reset_index()
    cat_agg['Category CTR'] = np.where(cat_agg['Views'] > 0, cat_agg['Clicks'] / cat_agg['Views'], 0)

    top_items = df_prod.groupby('Clean_Name').agg({
        'cat_m': 'first', 
        'Curr_Price': 'mean', 
        'Views': 'sum', 
        'Clicks': 'sum'
    }).reset_index()
    top_items.rename(columns={'Clean_Name': 'Product Name', 'cat_m': 'Category', 'Curr_Price': 'Price'}, inplace=True)
    top_items['Item CTR'] = np.where(top_items['Views'] > 0, top_items['Clicks'] / top_items['Views'], 0)

    reg_cat_agg = df_prod.groupby(['cat_m', 'Region']).agg({'Views': 'sum', 'Clicks': 'sum'}).reset_index()
    reg_cat_agg['CTR'] = np.where(reg_cat_agg['Views'] > 0, reg_cat_agg['Clicks'] / reg_cat_agg['Views'], 0)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not cat_agg.empty:
            cat_agg.sort_values(by='Category CTR', ascending=False).to_excel(writer, sheet_name='Top Categories', index=False)
        if not top_items.empty:
            top_items.sort_values(by='Item CTR', ascending=False).to_excel(writer, sheet_name='Top Items by CTR', index=False)
            top_items.sort_values(by='Clicks', ascending=False).to_excel(writer, sheet_name='Top Items by Clicks', index=False)
        if not reg_cat_agg.empty:
            pivot_reg_export = reg_cat_agg.pivot(index='cat_m', columns='Region', values='CTR').fillna(0).reset_index()
            pivot_reg_export.to_excel(writer, sheet_name='Category CTR by Region', index=False)
        
        reg_items_full = df_prod.groupby(['Region', 'Clean_Name']).agg({'cat_m': 'first', 'Views': 'sum', 'Clicks': 'sum'}).reset_index()
        reg_items_full.rename(columns={'Clean_Name': 'Product Name', 'cat_m': 'Category'}, inplace=True)
        reg_items_full['Item CTR'] = np.where(reg_items_full['Views'] > 0, reg_items_full['Clicks'] / reg_items_full['Views'], 0)
        reg_items_full = reg_items_full.sort_values(by=['Region', 'Item CTR'], ascending=[True, False])
        reg_items_full.to_excel(writer, sheet_name='All Items by Region', index=False)

    output.seek(0)
    dl_placeholder.download_button(
        label="⬇️ Download Regional Dashboard Report (.xlsx)",
        data=output,
        file_name="Regional_Campaign_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    merchant = "Grocery Outlet"
    macro_run_col = next((c for c in df_clean.columns if 'flyer run name' in str(c).lower() or 'campaign name' in str(c).lower()), None)
    runs_display = ", ".join([str(x) for x in df_clean[macro_run_col].dropna().unique()]) if macro_run_col else "Unknown Campaign"

    date_from_col = next((c for c in df_clean.columns if 'available from' in str(c).lower() or 'start date' in str(c).lower()), None)
    date_to_col = next((c for c in df_clean.columns if 'available to' in str(c).lower() or 'end date' in str(c).lower()), None)

    date_from = pd.to_datetime(df_clean[date_from_col], errors='coerce').min().strftime('%b %d, %Y') if date_from_col else "N/A"
    date_to = pd.to_datetime(df_clean[date_to_col], errors='coerce').max().strftime('%b %d, %Y') if date_to_col else "N/A"

    st.info(f"📍 **REGIONAL FLIGHT RECAP:** {merchant}  |  **Flyer Run Name(s):** {runs_display}  |  **Window:** {date_from} to {date_to}")
    st.write("---")

    st.subheader("📊 Top Categories by Shopper Engagement")
    
    col_cat_graph, col_cat_table = st.columns([7, 3])
    
    with col_cat_graph:
        max_cat_ctr = cat_agg['Category CTR'].max() if not cat_agg.empty else 0
        fig_cat = px.bar(cat_agg.sort_values(by='Category CTR', ascending=False).head(15), x='cat_m', y='Category CTR', color_discrete_sequence=['#43c4f4'])
        fig_cat.update_layout(
            title=dict(text='Top Categories by Shopper Engagement', x=0.5, xanchor='center', xref='paper', font=dict(family='Arial', size=16)),
            yaxis=dict(title="Item CTR", tickformat='.2%', dtick=0.005, range=[0, max_cat_ctr + 0.005]), 
            xaxis=dict(title=None)
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with col_cat_table:
        st.markdown("**🔍 Category Mapping Audit**")
        st.info("Cross-reference products against their newly assigned engine categories.")
        audit_df = df_prod[['Clean_Name', 'cat_m']].drop_duplicates().sort_values(by='Clean_Name').rename(columns={'Clean_Name': 'Name', 'cat_m': 'Assigned Category'}).reset_index(drop=True)
        st.dataframe(audit_df, use_container_width=True, hide_index=True, height=400)

    st.write("---")

    global_total_views = top_items['Views'].sum()
    global_total_clicks = top_items['Clicks'].sum()
    global_avg_ctr = global_total_clicks / global_total_views if global_total_views > 0 else 0
    global_avg_clicks = top_items['Clicks'].mean() if not top_items.empty else 0

    st.subheader("🏆 Top 10 Items - Shopper Interest by Item CTR")
    top_10_ctr = top_items.sort_values(by='Item CTR', ascending=False).head(10)
    st.metric(label="Avg. Item CTR (Global Campaign Baseline)", value=f"{global_avg_ctr:.2%}")
    st.dataframe(top_10_ctr[['Product Name', 'Category', 'Price', 'Views', 'Clicks', 'Item CTR']].style.format({'Price': '${:.2f}', 'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)

    st.write("---")

    st.subheader("🏆 Top 10 Items - Shopper Interest by Clicks")
    top_10_clicks = top_items.sort_values(by='Clicks', ascending=False).head(10)
    st.metric(label="Avg. Item Clicks (Global Campaign Baseline)", value=f"{global_avg_clicks:,.0f}")
    st.dataframe(top_10_clicks[['Product Name', 'Category', 'Price', 'Views', 'Clicks', 'Item CTR']].style.format({'Price': '${:.2f}', 'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)

    st.write("---")

    st.subheader("🗺️ Category Engagement by Region")
    if not reg_cat_agg.empty:
        col_tbl, col_chart = st.columns(2)
        pivot_reg = reg_cat_agg.pivot(index='cat_m', columns='Region', values='CTR').fillna(0)

        with col_tbl:
            st.markdown("<br>", unsafe_allow_html=True) 
            st.dataframe(pivot_reg.style.format('{:.2%}'), use_container_width=True)

        with col_chart:
            max_reg_ctr = reg_cat_agg['CTR'].max()
            color_map = {'East': '#00b050', 'West': '#073763', 'Nevada': '#43c4f4', 'Northwest': '#ffaf15', 'Other': '#94a3b8'}
            fig_reg = px.bar(reg_cat_agg, x='cat_m', y='CTR', color='Region', barmode='group', color_discrete_map=color_map)
            fig_reg.update_layout(
                title=dict(text='Category Engagement by Region', x=0.5, xanchor='center', xref='paper', font=dict(family='Arial', size=16)),
                yaxis=dict(title="Item CTR", tickformat='.2%', dtick=0.005, range=[0, max_reg_ctr + 0.005]),
                xaxis=dict(title=None)
            )
            st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("No regional category trends found.")

    st.write("---")
    st.subheader("📍 Top 5 Items by Region & Item CTR")

    flyer_runs = df_prod['Flyer Run Name'].dropna().unique()

    if len(flyer_runs) == 0:
        st.info("⚠️ No 'Flyer Run Name' data found in the upload.")
    else:
        for run in flyer_runs:
            st.markdown(f"#### 📅 Flyer Run: {run}")
            
            df_run = df_prod[df_prod['Flyer Run Name'] == run].copy()
            unique_regions = [r for r in df_run['Region'].unique() if pd.notna(r) and r != 'Other']

            if unique_regions:
                tab_reg = st.tabs(list(unique_regions))
                for i, r in enumerate(unique_regions):
                    with tab_reg[i]:
                        reg_items = df_run[df_run['Region'] == r].groupby('Clean_Name').agg({'Views': 'sum', 'Clicks': 'sum'}).reset_index()
                        reg_items.rename(columns={'Clean_Name': 'Product Name'}, inplace=True)
                        reg_items['Item CTR'] = np.where(reg_items['Views'] > 0, reg_items['Clicks'] / reg_items['Views'], 0)
                        reg_items = reg_items.sort_values(by=['Item CTR', 'Clicks'], ascending=[False, False]).head(5)
                        
                        st.dataframe(reg_items.style.format({'Views': '{:,.0f}', 'Clicks': '{:,.0f}', 'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)
            else:
                st.info(f"No localized regional items captured for {run}.")
                
            st.divider()

    st.write("---")
    st.subheader("🏆 Top 10 Items by Clicks & CTR (Per Flyer Run)")

    flyer_runs = df_prod['Flyer Run Name'].dropna().unique()

    if len(flyer_runs) == 0:
        st.info("⚠️ No 'Flyer Run Name' data found in the upload.")
    else:
        for run in flyer_runs:
            st.markdown(f"#### 📅 Flyer Run: {run}")
            
            df_run = df_prod[df_prod['Flyer Run Name'] == run].copy()
            item_stats = df_run.groupby('Clean_Name').agg({'Views': 'sum', 'Clicks': 'sum'}).reset_index()
            item_stats['Item CTR'] = np.where(item_stats['Views'] > 0, item_stats['Clicks'] / item_stats['Views'], 0)
            item_stats.rename(columns={'Clean_Name': 'Merchandise Name', 'Clicks': 'Total Clicks'}, inplace=True)
            
            top_10_clicks = item_stats.sort_values(by='Total Clicks', ascending=False).head(10)[['Merchandise Name', 'Total Clicks']]
            top_10_ctr = item_stats.sort_values(by=['Item CTR', 'Total Clicks'], ascending=[False, False]).head(10)[['Merchandise Name', 'Item CTR']]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔥 Top 10 by Total Clicks**")
                st.dataframe(top_10_clicks.style.format({'Total Clicks': '{:,.0f}'}), use_container_width=True, hide_index=True)
                
            with col2:
                st.markdown("**🎯 Top 10 by Item CTR**")
                st.dataframe(top_10_ctr.style.format({'Item CTR': '{:.2%}'}), use_container_width=True, hide_index=True)
                
            st.divider()

# ---------------------------------------------------------
# 🧭 SIDEBAR NAVIGATION MENU
# ---------------------------------------------------------
pipeline_mode = st.sidebar.radio(
    "Select Analysis Module:",
    [
        "🗂️ Module 1: Campaign Performance Breakdown", 
        "⚖️ Module 2: Head-to-Head Comparison", 
        "🏆 Module 3: Industry Benchmarks", 
        "🛒 Module 4: Taylor's Workspace"
    ]
)

# ---------------------------------------------------------
# 🚦 MODULE ROUTER (MUST MATCH MENU STRINGS ABOVE)
# ---------------------------------------------------------
if "Campaign Performance Breakdown" in pipeline_mode or "Module 1" in pipeline_mode or "Single Campaign" in pipeline_mode:
    render_single_campaign_matrix()
elif "Head-to-Head" in pipeline_mode or "Module 2" in pipeline_mode:
    render_head_to_head_variance()
elif "Industry Benchmarks" in pipeline_mode or "Module 3" in pipeline_mode:
    render_benchmark_scorecard()
elif "Taylor" in pipeline_mode or "Module 4" in pipeline_mode:
    render_taylors_workspace()
