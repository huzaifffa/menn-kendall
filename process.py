import pandas as pd
from io import BytesIO
import re

def parse_excel_content_to_dataframes(file_content):
    """
    Parse an actual .xlsx file into pandas DataFrames for specific sheets.
    Accepts bytes (file content) or a file path.
    """
    excel_source = BytesIO(file_content) if isinstance(file_content, (bytes, bytearray)) else file_content

    # Read the Excel file using pandas with openpyxl engine
    try:
        xls = pd.ExcelFile(excel_source, engine='openpyxl')
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file: {e}. If this is an .xlsx file, please ensure 'openpyxl' is installed.")

    target_sheets = ['Chart', 'Queries', 'Pages']

    dataframes = {}
    for sheet in target_sheets:
        if sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
            # Normalize column names (strip whitespace)
            df.columns = [str(c).strip() for c in df.columns]
            # Convert known numeric columns if present
            for col in ['Clicks', 'Impressions', 'CTR', 'Position']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Convert Date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            dataframes[sheet] = df
        else:
            dataframes[sheet] = None

    return dataframes

# Load your file content
with open('data.xlsx', 'rb') as file:
    file_content = file.read()

# Parse the content into DataFrames
dfs = parse_excel_content_to_dataframes(file_content)

# Access the individual DataFrames
chart_df = dfs.get('Chart')
queries_df = dfs.get('Queries')
pages_df = dfs.get('Pages')

# Show each parsed DataFrame
for name, df in [('Chart', chart_df), ('Queries', queries_df), ('Pages', pages_df)]:
    print("=" * 60)
    print(f"{name.upper()} DataFrame")
    print("=" * 60)
    if df is None:
        print(f"Sheet '{name}' not found.")
    else:
        print(df.head(20))
        print(f"\nShape: {df.shape}\nColumns: {list(df.columns)}")
    print("\n")

# Guard against missing/failed parsing
missing = [name for name, df in [('Chart', chart_df), ('Queries', queries_df), ('Pages', pages_df)] if df is None]
if missing:
    print("Warning: Failed to parse sheets:", ", ".join(missing))

# Display information about each DataFrame if available
if chart_df is not None:
    print("=" * 60)
    print("CHART DataFrame Info:")
    print("=" * 60)
    print(f"Shape: {chart_df.shape}")
    print(f"Columns: {list(chart_df.columns)}")
    if 'Date' in chart_df.columns:
        print(f"Date Range: {chart_df['Date'].min()} to {chart_df['Date'].max()}")
    print("\nFirst 5 rows:")
    print(chart_df.head())
    print("\n" + "=" * 60)

if queries_df is not None:
    print("\nQUERIES DataFrame Info:")
    print("=" * 60)
    print(f"Shape: {queries_df.shape}")
    print(f"Columns: {list(queries_df.columns)}")
    if 'Top queries' in queries_df.columns:
        print(f"Total unique queries: {queries_df['Top queries'].nunique()}")
    if 'Clicks' in queries_df.columns:
        print(f"Total clicks: {queries_df['Clicks'].sum():.0f}")
    if 'Impressions' in queries_df.columns:
        print(f"Total impressions: {queries_df['Impressions'].sum():.0f}")
    if {'Top queries','Clicks','Impressions','CTR','Position'}.issubset(set(queries_df.columns)):
        print("\nTop 10 queries by clicks:")
        top_queries = queries_df.sort_values('Clicks', ascending=False).head(10)
        print(top_queries[['Top queries', 'Clicks', 'Impressions', 'CTR', 'Position']])
    print("\n" + "=" * 60)

if pages_df is not None:
    print("\nPAGES DataFrame Info:")
    print("=" * 60)
    print(f"Shape: {pages_df.shape}")
    print(f"Columns: {list(pages_df.columns)}")
    if 'Top pages' in pages_df.columns:
        print(f"Total pages: {pages_df['Top pages'].nunique()}")
    if 'Clicks' in pages_df.columns:
        print(f"Total clicks: {pages_df['Clicks'].sum():.0f}")
    if 'Impressions' in pages_df.columns:
        print(f"Total impressions: {pages_df['Impressions'].sum():.0f}")
    if {'Top pages','Clicks','Impressions','CTR','Position'}.issubset(set(pages_df.columns)):
        print("\nTop 10 pages by clicks:")
        top_pages = pages_df.sort_values('Clicks', ascending=False).head(10)
        print(top_pages[['Top pages', 'Clicks', 'Impressions', 'CTR', 'Position']])
    print("\n" + "=" * 60)

# Additional analysis functions
def analyze_seo_performance(chart_df, queries_df, pages_df):
    """Perform basic SEO performance analysis"""
    
    print("\n" + "=" * 60)
    print("SEO PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if chart_df is not None and {'CTR','Position','Clicks','Impressions'}.issubset(set(chart_df.columns)):
        avg_ctr = chart_df['CTR'].mean()
        avg_position = chart_df['Position'].mean()
        total_clicks = chart_df['Clicks'].sum()
        total_impressions = chart_df['Impressions'].sum()
        
        print(f"Overall Average CTR: {avg_ctr:.4f}")
        print(f"Overall Average Position: {avg_position:.2f}")
        print(f"Total Clicks: {total_clicks:.0f}")
        print(f"Total Impressions: {total_impressions:.0f}")
    else:
        print("Chart data unavailable or missing required columns.")
    
    if queries_df is not None and {'CTR','Position','Top queries','Clicks'}.issubset(set(queries_df.columns)):
        high_ctr_queries = queries_df[queries_df['CTR'] > 0.1].sort_values('CTR', ascending=False)
        print(f"\nHigh CTR Queries (CTR > 0.1): {len(high_ctr_queries)}")
        if len(high_ctr_queries) > 0:
            print("Top 5 high CTR queries:")
            for idx, row in high_ctr_queries.head().iterrows():
                print(f"  - {row['Top queries']}: CTR={row['CTR']:.3f}, Position={row['Position']:.1f}")
        zero_click_queries = queries_df[queries_df['Clicks'] == 0]
        print(f"\nQueries with 0 Clicks: {len(zero_click_queries)}")
        if 'Impressions' in zero_click_queries.columns:
            print(f"Impressions lost on 0-click queries: {zero_click_queries['Impressions'].sum():.0f}")
    else:
        print("Queries data unavailable or missing required columns.")
    
    if pages_df is not None and {'CTR','Clicks','Top pages'}.issubset(set(pages_df.columns)):
        high_ctr_pages = pages_df[pages_df['CTR'] > 0.03].sort_values('CTR', ascending=False)
        print(f"\nHigh CTR Pages (CTR > 0.03): {len(high_ctr_pages)}")
        if len(high_ctr_pages) > 0:
            print("Top 5 high CTR pages:")
            for idx, row in high_ctr_pages.head().iterrows():
                page_name = row['Top pages'].split('/')[-2] if '/' in row['Top pages'] else row['Top pages']
                clicks_val = row['Clicks'] if 'Clicks' in high_ctr_pages.columns else float('nan')
                print(f"  - {page_name}: CTR={row['CTR']:.3f}, Clicks={clicks_val:.0f}")
    else:
        print("Pages data unavailable or missing required columns.")
    
    # Identify pages needing improvement based on heuristics
    if pages_df is not None and {'Top pages','Clicks','Impressions','CTR','Position'}.issubset(set(pages_df.columns)):
        print("\n" + "=" * 60)
        print("PAGES NEEDING IMPROVEMENT")
        print("=" * 60)
        
        dfp = pages_df.copy()
        # Replace NaNs
        dfp['Clicks'] = pd.to_numeric(dfp['Clicks'], errors='coerce').fillna(0)
        dfp['Impressions'] = pd.to_numeric(dfp['Impressions'], errors='coerce').fillna(0)
        dfp['CTR'] = pd.to_numeric(dfp['CTR'], errors='coerce').fillna(0)
        dfp['Position'] = pd.to_numeric(dfp['Position'], errors='coerce').fillna(0)
        
        # Heuristics:
        # 1) High impressions, low CTR
        hi_imp_low_ctr = dfp[(dfp['Impressions'] >= dfp['Impressions'].median()) & (dfp['CTR'] < 0.01)]
        # 2) Poor average position (>= 15) but non-trivial impressions
        poor_pos = dfp[(dfp['Position'] >= 15) & (dfp['Impressions'] >= dfp['Impressions'].median())]
        # 3) Zero clicks but impressions
        zero_clicks = dfp[(dfp['Clicks'] == 0) & (dfp['Impressions'] > 0)]
        
        # Combine and rank by impressions to prioritize impact
        import numpy as np
        candidates = pd.concat([hi_imp_low_ctr.assign(reason='High impressions, low CTR'),
                                poor_pos.assign(reason='Poor position, decent impressions'),
                                zero_clicks.assign(reason='Zero clicks with impressions')],
                               ignore_index=True)
        if not candidates.empty:
            candidates = candidates.sort_values(['Impressions'], ascending=False)
            candidates = candidates.drop_duplicates(subset=['Top pages'])
            show_cols = ['Top pages','Clicks','Impressions','CTR','Position','reason']
            print("Top pages to improve (prioritized by impressions):")
            print(candidates[show_cols].head(10))
            print("\nRecommendations:")
            print("- For high impressions but low CTR: improve titles/meta descriptions, add clear CTAs, align snippet to intent.")
            print("- For poor position: strengthen internal links, add relevant content, target better keywords, consider backlinks.")
            print("- For zero clicks with impressions: ensure compelling snippet, add structured data, check page relevance for shown queries.")
        else:
            print("No pages meet the improvement criteria.")
        
        # Also show pages that do NOT need improvement (good performers)
        print("\n" + "=" * 60)
        print("PAGES NOT NEEDING IMPROVEMENT")
        print("=" * 60)
        # Always surface top-performing pages using a composite score
        dfp_perf = dfp.copy()
        # Safe numerics
        for c in ['Clicks','Impressions','CTR','Position']:
            dfp_perf[c] = pd.to_numeric(dfp_perf[c], errors='coerce').fillna(0)
        # Build a composite score: favor high CTR, high clicks, high impressions, low position
        import numpy as np
        # Avoid division by zero
        pos_component = 1 / (dfp_perf['Position'].replace(0, np.nan)).fillna(dfp_perf['Position'].replace(0, np.nan).median())
        # Normalize components roughly
        ctr_comp = dfp_perf['CTR']
        clicks_comp = dfp_perf['Clicks'] / (dfp_perf['Clicks'].max() or 1)
        imp_comp = dfp_perf['Impressions'] / (dfp_perf['Impressions'].max() or 1)
        score = (0.4 * ctr_comp) + (0.3 * clicks_comp) + (0.2 * imp_comp) + (0.1 * pos_component)
        dfp_perf['performance_score'] = score
        top_good = dfp_perf.sort_values('performance_score', ascending=False).head(10)
        print(top_good[['Top pages','Clicks','Impressions','CTR','Position','performance_score']])
        
        # Define good performance criteria (slightly relaxed to surface results)
        imp_thresh = dfp['Impressions'].median()
        ctr_thresh = 0.02  # previously 0.03
        pos_thresh = 12    # previously 10
        good_pages = dfp[(dfp['Impressions'] >= imp_thresh) & (dfp['CTR'] >= ctr_thresh) & (dfp['Position'] <= pos_thresh) & (dfp['Clicks'] > 0)]
        if not good_pages.empty:
            good_pages = good_pages.sort_values(['Clicks','CTR'], ascending=[False, False])
            print(good_pages[['Top pages','Clicks','Impressions','CTR','Position']].head(15))
        else:
            # If none match, show top pages by CTR with decent impressions as a fallback
            fallback = dfp[dfp['Impressions'] >= imp_thresh].sort_values('CTR', ascending=False).head(10)
            if not fallback.empty:
                print("No pages meet the strict good performance criteria; showing top CTR pages with decent impressions:")
                print(fallback[['Top pages','Clicks','Impressions','CTR','Position']])
            else:
                print("No pages meet the good performance criteria.")

# Run analysis
analyze_seo_performance(chart_df, queries_df, pages_df)