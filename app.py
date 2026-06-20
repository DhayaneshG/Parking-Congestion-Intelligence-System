# ============================================================
# PCIS — Parking Congestion Intelligence System
# Flipkart Gridlock 2.0 x Bengaluru Traffic Police
# Direction C — Urban Heat Map
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="PCIS — Parking Congestion Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0d1117; }

section[data-testid="stSidebar"] { background-color: #0a0d12 !important; border-right: 1px solid #1e2530; }

.stTabs [data-baseweb="tab-list"] { background-color: #0d1117; border-bottom: 1px solid #1e2530; gap: 0; }
.stTabs [data-baseweb="tab"] { background-color: transparent; color: #4a5568; font-size: 12px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; padding: 12px 22px; border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"] { color: #e07b39 !important; border-bottom: 2px solid #e07b39 !important; background-color: transparent !important; }

.kpi-card { background: #111720; border: 1px solid #1e2530; border-radius: 6px; padding: 18px 20px; }
.kpi-label { font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #4a5568; margin-bottom: 10px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 500; color: #e2e8f0; line-height: 1; }
.kpi-value.critical { color: #e05252; }
.kpi-value.high     { color: #e07b39; }
.kpi-value.teal     { color: #38a89d; }
.kpi-value.amber    { color: #d4a843; }

.section-label { font-size: 10px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #6b7280; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #1e2530; }

.stDownloadButton button { background: transparent !important; border: 1px solid #2d3748 !important; color: #8b95a1 !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
.stDownloadButton button:hover { border-color: #e07b39 !important; color: #e07b39 !important; }
</style>
""", unsafe_allow_html=True)

TIER_COLORS = {
    'CRITICAL': '#e05252',
    'HIGH'    : '#e07b39',
    'MEDIUM'  : '#d4a843',
    'LOW'     : '#38a89d'
}

REDUCTION_MAP = {
    'CRITICAL': 0.60,
    'HIGH'    : 0.50,
    'MEDIUM'  : 0.40,
    'LOW'     : 0.30
}

# ── DATA LOADING ──────────────────────────────────────────────
@st.cache_data
def load_data():
    pcis   = pd.read_csv('stage4_pcis_scores.csv')
    patrol = pd.read_csv('stage6_patrol_plan.csv')
    prophet = pd.read_csv('stage5_prophet_summary.csv')
    raw    = pd.read_csv('stage1_cleaned.csv', parse_dates=['created_datetime'])
    raw    = raw.dropna(subset=['latitude', 'longitude'])
    raw    = raw[
        raw['latitude'].between(12.7, 13.2) &
        raw['longitude'].between(77.4, 77.8)
    ].reset_index(drop=True)
    # Normalize shift column — works regardless of what it was named
    for col in ['primary_shift', 'Shift', 'shift']:
        if col in patrol.columns:
            patrol['shift_label'] = patrol[col]
            break
    if 'shift_label' not in patrol.columns:
        patrol['shift_label'] = 'N/A'
    return pcis, patrol, prophet, raw

pcis, patrol, prophet, raw = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-size:11px;font-weight:700;letter-spacing:0.14em;'
        'text-transform:uppercase;color:#e07b39;margin:8px 0 2px 0">Parking Congestion Intelligence System</p>'
        '<p style="font-size:12px;color:#6b7280;margin:0 0 16px 0">'
        'PCIS tells Bengaluru Traffic Police exactly where to be, when to be there, and what happens to city-wide congestion when they arrive.</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<hr style="border:none;border-top:1px solid #1e2530;margin:16px 0">'
        '<p style="font-size:11px;font-weight:700;letter-spacing:0.12em;'
        'text-transform:uppercase;color:#e07b39;margin-bottom:8px">Team</p>'
        '<p style="font-size:14px;color:#c9d1d9;font-weight:500;margin:0 0 4px 0">The Perceptrons</p>'
        '<hr style="border:none;border-top:1px solid #1e2530;margin:0 0 16px 0">'
        '<p style="font-size:11px;font-weight:700;letter-spacing:0.12em;'
        'text-transform:uppercase;color:#e07b39;margin-bottom:8px">Institution</p>'
        '<p style="font-size:14px;color:#c9d1d9;font-weight:500;margin:0">Rajalakshmi Engineering College</p>'
         '<hr style="border:none;border-top:1px solid #1e2530;margin:0 0 16px 0">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p style="font-size:11px;font-weight:700;letter-spacing:0.12em;'
        'text-transform:uppercase;color:#e07b39;margin-bottom:8px">Severity Filter</p>',
        unsafe_allow_html=True
    )
    tier_filter = st.multiselect(
        "Severity",
        options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
        default=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
        label_visibility="collapsed"
    )

    st.markdown(
        '<hr style="border:none;border-top:1px solid #1e2530;margin:16px 0">'
        '<p style="font-size:11px;font-weight:700;letter-spacing:0.12em;'
        'text-transform:uppercase;color:#e07b39;margin-bottom:8px">Patrol Deployment</p>',
        unsafe_allow_html=True
    )
    n_units = st.slider("Patrol Units to Deploy", min_value=1, max_value=20, value=10)

# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div style="padding:28px 0 22px 0;border-bottom:1px solid #1e2530;margin-bottom:24px">
    <p style="font-size:32px;font-weight:700;color:#f0f6fc;letter-spacing:-0.03em;
       line-height:1.1;margin:0 0 6px 0">
       Parking Congestion Intelligence System
    </p>
    <p style="font-size:13px;color:#e07b39;font-weight:600;letter-spacing:0.08em;
       text-transform:uppercase;margin:0 0 16px 0">
       Flipkart Gridlock 2.0 &nbsp;·&nbsp; Bengaluru Traffic Police
    </p>
    <p style="font-size:13px;color:#6b7280;font-weight:400;margin:0 0 16px 0;
       letter-spacing:0.01em;line-height:1.7;max-width:860px">
       Built on 298,450 real BTP violation records — PCIS detects the city's highest-impact
       parking hotspots, scores their congestion severity, forecasts when they will peak,
       and tells commanders exactly which roads to patrol and when.
    </p>
    <div style="display:flex;gap:32px;flex-wrap:wrap">
        <div>
            <p style="font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500;
               color:#e05252;margin:0 0 2px 0">20</p>
            <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;
               text-transform:uppercase;color:#4a5568;margin:0">Patrol Units</p>
        </div>
        <div style="width:1px;background:#1e2530"></div>
        <div>
            <p style="font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500;
               color:#d4a843;margin:0 0 2px 0">279</p>
            <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;
               text-transform:uppercase;color:#4a5568;margin:0">Hotspots Covered</p>
        </div>
        <div style="width:1px;background:#1e2530"></div>
        <div>
            <p style="font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500;
               color:#38a89d;margin:0 0 2px 0">14.5%</p>
            <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;
               text-transform:uppercase;color:#4a5568;margin:0">PCIS Reduction</p>
        </div>
        <div style="width:1px;background:#1e2530"></div>
        <div>
            <p style="font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500;
               color:#e07b39;margin:0 0 2px 0">0.86</p>
            <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;
               text-transform:uppercase;color:#4a5568;margin:0">Model R²</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
# ── KPI BAR ───────────────────────────────────────────────────
tier_counts = pcis['severity_tier'].value_counts()
top_pcis    = pcis['PCIS_score'].max()
total_viols = pcis['violation_count'].sum()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Hotspot Clusters</div>'
                f'<div class="kpi-value">{len(pcis)}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Critical Zones</div>'
                f'<div class="kpi-value critical">{tier_counts.get("CRITICAL", 0)}</div></div>',
                unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">High Risk Zones</div>'
                f'<div class="kpi-value high">{tier_counts.get("HIGH", 0)}</div></div>',
                unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Top PCIS Score</div>'
                f'<div class="kpi-value amber">{top_pcis:.0f}</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Violations Mapped</div>'
                f'<div class="kpi-value teal">{total_viols:,}</div></div>', unsafe_allow_html=True)

st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Violation Heatmap",
    "Patrol Planner",
    "Temporal Analysis",
    "Hotspot Explorer",
    "Model Intelligence"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — VIOLATION HEATMAP
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-label">Spatial distribution of parking violations across Bengaluru</p>',
                unsafe_allow_html=True)

    col_map, col_ctrl = st.columns([4, 1])

    with col_ctrl:
        st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:0.1em;'
                    'text-transform:uppercase;color:#6b7280;margin-bottom:8px">Map Layer</p>',
                    unsafe_allow_html=True)
        map_layer   = st.radio("Map Layer", ["Heatmap", "Cluster Markers", "Both"],
                               label_visibility="collapsed")
        show_patrol = st.checkbox("Show patrol positions", value=True)

        st.markdown('<hr style="border:none;border-top:1px solid #1e2530;margin:12px 0">'
                    '<p style="font-size:10px;font-weight:700;letter-spacing:0.1em;'
                    'text-transform:uppercase;color:#6b7280;margin-bottom:10px">Severity</p>',
                    unsafe_allow_html=True)
        for tier, color in TIER_COLORS.items():
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0"></div>'
                f'<span style="font-size:12px;color:#8b95a1">{tier}</span></div>',
                unsafe_allow_html=True
            )

    with col_map:
        filtered_pcis = pcis[pcis['severity_tier'].isin(tier_filter)]

        m = folium.Map(
            location=[12.97, 77.59],
            zoom_start=12,
            tiles='CartoDB dark_matter'
        )

        if map_layer in ["Heatmap", "Both"]:
            heat_data = [
                [r['lat_center'], r['lon_center'], r['PCIS_score']]
                for _, r in filtered_pcis.iterrows()
            ]
            HeatMap(
                heat_data,
                min_opacity=0.35,
                max_zoom=15,
                radius=20,
                blur=14,
                gradient={
                    0.0: '#0d1117',
                    0.3: '#1a3a2a',
                    0.5: '#d4a843',
                    0.75: '#e07b39',
                    1.0: '#e05252'
                }
            ).add_to(m)

        if map_layer in ["Cluster Markers", "Both"]:
            mc = MarkerCluster(name="Hotspots")
            for _, row in filtered_pcis.iterrows():
                color = TIER_COLORS.get(row['severity_tier'], '#888')
                folium.CircleMarker(
                    location=[row['lat_center'], row['lon_center']],
                    radius=max(4, min(16, row['PCIS_score'] / 8)),
                    color=color, fill=True, fill_opacity=0.8, weight=1,
                    popup=folium.Popup(
                        f"<div style='font-family:monospace;font-size:12px;line-height:1.6'>"
                        f"<b style='color:{color}'>#{int(row['pcis_rank'])} {row['road_name']}</b><br>"
                        f"PCIS Score &nbsp;{row['PCIS_score']:.1f} / 100<br>"
                        f"Tier &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{row['severity_tier']}<br>"
                        f"Violations &nbsp;{int(row['violation_count']):,}<br>"
                        f"Road type &nbsp;{row['highway_type']}<br>"
                        f"Peak ratio &nbsp;{row['peak_ratio']:.1%}<br>"
                        f"Peak hour &nbsp;&nbsp;{int(row['dominant_hour'])}:00</div>",
                        max_width=240
                    )
                ).add_to(mc)
            mc.add_to(m)

        if show_patrol:
            for _, p in patrol.iterrows():
                folium.Marker(
                    location=[p['lat'], p['lon']],
                    icon=folium.Icon(color='blue', icon='map-marker', prefix='fa'),
                    popup=folium.Popup(
                        f"<div style='font-family:monospace;font-size:12px;line-height:1.6'>"
                        f"<b>{p['unit_id']}</b><br>"
                        f"{p['road_name']}<br>"
                        f"Shift &nbsp;&nbsp;&nbsp;{p['shift_label']}<br>"
                        f"Coverage &nbsp;{int(p['hotspots_covered'])} hotspots</div>",
                        max_width=200
                    )
                ).add_to(m)

        st_folium(m, width=None, height=560, returned_objects=[])

# ══════════════════════════════════════════════════════════════
# TAB 2 — PATROL PLANNER
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">resource allocation — maximise PCIS reduction per deployed unit</p>',
                unsafe_allow_html=True)

    def classify_shift(hour):
        h = int(hour)
        if 6 <= h < 10:   return 'Morning'
        elif 10 <= h < 17: return 'Afternoon'
        elif 17 <= h < 22: return 'Evening'
        else:              return 'Night'

    coords_p    = pcis[['lat_center', 'lon_center']].values
    tree_p      = cKDTree(coords_p * np.array([111.0, 111.0]))
    pcis_sorted = pcis.sort_values('PCIS_score', ascending=False).copy()
    assigned, plan, uid = set(), [], 1

    for _, row in pcis_sorted.iterrows():
        if uid > n_units:
            break
        if row.name in assigned:
            continue
        pt     = np.array([row['lat_center'] * 111.0, row['lon_center'] * 111.0])
        nearby = tree_p.query_ball_point(pt, r=2.0)
        assigned.update(nearby)
        rf     = REDUCTION_MAP.get(row['severity_tier'], 0.40)
        plan.append({
            'Unit'         : f"BTP-{uid:02d}",
            'Road'         : row['road_name'],
            'Type'         : row['highway_type'],
            'PCIS'         : round(row['PCIS_score'], 1),
            'Tier'         : row['severity_tier'],
            'Shift'        : classify_shift(row['dominant_hour']),
            'Priority Day' : row['dominant_day'],
            'Covers'       : len(nearby),
            'PCIS Reduced' : round(row['PCIS_score'] * rf, 1)
        })
        uid += 1

    plan_df = pd.DataFrame(plan)

    covered = set()
    for _, r in plan_df.iterrows():
        match = pcis[pcis['road_name'] == r['Road']]
        if len(match):
            pt = np.array([match.iloc[0]['lat_center'] * 111.0,
                           match.iloc[0]['lon_center'] * 111.0])
            covered.update(tree_p.query_ball_point(pt, r=2.0))

    total_before = pcis['PCIS_score'].sum()
    pcis_sim     = pcis.copy()
    for i in covered:
        if i < len(pcis_sim):
            tier = pcis_sim.iloc[i]['severity_tier']
            pcis_sim.iloc[i, pcis_sim.columns.get_loc('PCIS_score')] *= (
                1 - REDUCTION_MAP.get(tier, 0.40))
    total_after = pcis_sim['PCIS_score'].sum()
    reduction   = 100 * (total_before - total_after) / total_before

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Units Deployed</div>'
                    f'<div class="kpi-value">{n_units}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Hotspots Covered</div>'
                    f'<div class="kpi-value teal">{len(covered)}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">City PCIS Before</div>'
                    f'<div class="kpi-value critical">{total_before:.0f}</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">City PCIS After</div>'
                    f'<div class="kpi-value teal">{total_after:.0f} '
                    f'<span style="font-size:14px">-{reduction:.1f}%</span></div></div>',
                    unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Patrol assignment table</p>', unsafe_allow_html=True)
    st.dataframe(plan_df, use_container_width=True, height=340)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">PCIS before vs after deployment — top 50 hotspots</p>',
                unsafe_allow_html=True)

    top50_b = pcis.sort_values('PCIS_score', ascending=False).head(50)['PCIS_score'].values
    top50_a = pcis_sim.sort_values('PCIS_score', ascending=False).head(50)['PCIS_score'].values
    x       = np.arange(len(top50_b))

    fig_sim, ax_sim = plt.subplots(figsize=(12, 3.5))
    fig_sim.patch.set_facecolor('#0d1117')
    ax_sim.set_facecolor('#111720')
    ax_sim.bar(x, top50_b, color='#e05252', alpha=0.55, width=0.8, label='Before patrol')
    ax_sim.bar(x, top50_a, color='#38a89d', alpha=0.85, width=0.8, label='After patrol')
    ax_sim.set_xlabel('Hotspot rank', color='#4a5568', fontsize=10)
    ax_sim.set_ylabel('PCIS score',   color='#4a5568', fontsize=10)
    ax_sim.tick_params(colors='#4a5568', labelsize=9)
    ax_sim.spines[['top', 'right', 'bottom', 'left']].set_color('#1e2530')
    ax_sim.legend(facecolor='#111720', labelcolor='#8b95a1',
                  edgecolor='#1e2530', fontsize=9)
    fig_sim.tight_layout()
    st.pyplot(fig_sim)

# ══════════════════════════════════════════════════════════════
# TAB 3 — TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">Violation patterns by hour, day and 30-day forecast</p>',
                unsafe_allow_html=True)

    PEAK_HOURS = list(range(7, 10)) + list(range(17, 22))

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown('<p class="section-label">Violations by hour of day</p>', unsafe_allow_html=True)
        hourly = raw.groupby('hour').size().reset_index(name='count')
        fig_h, ax_h = plt.subplots(figsize=(7, 3.5))
        fig_h.patch.set_facecolor('#0d1117')
        ax_h.set_facecolor('#111720')
        colors_h = ['#e07b39' if h in PEAK_HOURS else '#1e3040' for h in hourly['hour']]
        ax_h.bar(hourly['hour'], hourly['count'], color=colors_h,
                 edgecolor='#0d1117', linewidth=0.5, width=0.85)
        ax_h.set_xlabel('Hour', color='#4a5568', fontsize=9)
        ax_h.set_ylabel('Violations', color='#4a5568', fontsize=9)
        ax_h.set_xticks(range(0, 24))
        ax_h.tick_params(colors='#4a5568', labelsize=8)
        ax_h.spines[['top', 'right', 'bottom', 'left']].set_color('#1e2530')
        orange_p = mpatches.Patch(color='#e07b39', label='Peak window')
        dark_p   = mpatches.Patch(color='#1e3040', label='Off-peak')
        ax_h.legend(handles=[orange_p, dark_p], facecolor='#111720',
                    labelcolor='#8b95a1', edgecolor='#1e2530', fontsize=8)
        fig_h.tight_layout()
        st.pyplot(fig_h)

    with col_t2:
        st.markdown('<p class="section-label">Violations by day of week</p>', unsafe_allow_html=True)
        daily_v = raw.groupby('day_of_week').size().reset_index(name='count')
        fig_d, ax_d = plt.subplots(figsize=(7, 3.5))
        fig_d.patch.set_facecolor('#0d1117')
        ax_d.set_facecolor('#111720')
        ax_d.bar(daily_v['day_of_week'], daily_v['count'],
                 color='#1e3040', edgecolor='#0d1117', linewidth=0.5, width=0.75)
        ax_d.set_xlabel('Day', color='#4a5568', fontsize=9)
        ax_d.set_ylabel('Violations', color='#4a5568', fontsize=9)
        ax_d.tick_params(colors='#4a5568', labelsize=8)
        ax_d.spines[['top', 'right', 'bottom', 'left']].set_color('#1e2530')
        fig_d.tight_layout()
        st.pyplot(fig_d)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">30-day peak forecast — top 20 hotspots by PCIS</p>',
                unsafe_allow_html=True)

    prophet_disp = prophet.sort_values('pcis_rank')[[
        'pcis_rank', 'road_name', 'PCIS_score', 'peak_forecast_30d'
    ]].rename(columns={
        'pcis_rank'        : 'Rank',
        'road_name'        : 'Road',
        'PCIS_score'       : 'PCIS Score',
        'peak_forecast_30d': 'Peak Forecast (30 days)'
    })
    st.dataframe(prophet_disp, use_container_width=True, height=420)

# ══════════════════════════════════════════════════════════════
# TAB 4 — HOTSPOT EXPLORER
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-label">Filter and inspect individual hotspot clusters</p>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        tier_sel = st.multiselect(
            "Severity tier",
            ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
            default=['CRITICAL', 'HIGH']
        )
    with f2:
        road_sel = st.multiselect(
            "Road type",
            sorted(pcis['highway_type'].unique().tolist()),
            default=sorted(pcis['highway_type'].unique().tolist())
        )
    with f3:
        min_v = int(pcis['violation_count'].min())
        max_v = int(pcis['violation_count'].max())
        vrange = st.slider("Violation count", min_v, max_v, (min_v, max_v))

    filtered = pcis[
        pcis['severity_tier'].isin(tier_sel) &
        pcis['highway_type'].isin(road_sel) &
        pcis['violation_count'].between(*vrange)
    ].sort_values('PCIS_score', ascending=False)

    st.markdown(
        f'<p style="font-size:11px;color:#4a5568;margin:12px 0 8px 0">'
        f'{len(filtered)} hotspots match current filters</p>',
        unsafe_allow_html=True
    )

    display_cols = [
        'pcis_rank', 'road_name', 'highway_type', 'violation_count',
        'peak_ratio', 'dominant_hour', 'dominant_day',
        'road_capacity_factor', 'PCIS_score', 'severity_tier'
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, height=480)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.download_button(
        label="Download filtered results as CSV",
        data=filtered[display_cols].to_csv(index=False),
        file_name="pcis_hotspots_filtered.csv",
        mime="text/csv"
    )

    # ══════════════════════════════════════════════════════════════
# TAB 5 — MODEL INTELLIGENCE
# ══════════════════════════════════════════════════════════════
with tab5:
    import os
    st.markdown('<p class="section-label">How the system learns — pipeline, clustering, scoring and prediction</p>',
                unsafe_allow_html=True)

    # ── SECTION 1: PIPELINE DIAGRAM ──────────────────────────
    st.markdown('<p class="section-label">7-stage intelligence pipeline</p>',
                unsafe_allow_html=True)

    stages = [
        ("01", "EDA",           "115,350 approved records\nParsed datetime, GPS, violations"),
        ("02", "DBSCAN",        "951 hotspot clusters\nε=50m · min_samples=5 · haversine"),
        ("03", "Road Network",  "Highway type per hotspot\nCapacity factor: 0.15 → 0.95"),
        ("04", "PCIS Formula",  "4-component impact score\nLog-normalized · 0–100 scale"),
        ("05", "Forecasting",   "Prophet: 30-day forecast\nXGBoost: R²=0.86"),
        ("06", "Patrol Engine", "Greedy unit allocation\n10 units · 14.5% reduction"),
        ("07", "Dashboard",     "Live heatmap + planner\nStreamlit · Folium · Dark UI"),
    ]

    fig_pipe, ax_pipe = plt.subplots(figsize=(14, 2.8))
    fig_pipe.patch.set_facecolor('#0d1117')
    ax_pipe.set_facecolor('#0d1117')
    ax_pipe.axis('off')

    box_w, box_h = 1.6, 1.4
    gap          = 0.42
    total_w      = len(stages) * box_w + (len(stages) - 1) * gap
    start_x      = (14 - total_w) / 2
    accent_colors = ['#e05252','#e07b39','#d4a843','#e07b39','#38a89d','#38a89d','#6b7280']

    for i, (num, title, desc) in enumerate(stages):
        x     = start_x + i * (box_w + gap)
        color = accent_colors[i]
        rect  = plt.Rectangle((x, 0.1), box_w, box_h,
                               facecolor='#111720', edgecolor=color,
                               linewidth=1.2, zorder=2)
        ax_pipe.add_patch(rect)
        ax_pipe.text(x + 0.12, 0.1 + box_h - 0.18, num,
                     fontsize=7, color=color, fontweight='700',
                     fontfamily='monospace', va='top', zorder=3)
        ax_pipe.text(x + box_w / 2, 0.1 + box_h - 0.38, title,
                     fontsize=8.5, color='#e2e8f0', fontweight='600',
                     ha='center', va='top', zorder=3)
        ax_pipe.text(x + box_w / 2, 0.1 + box_h - 0.68, desc,
                     fontsize=6.5, color='#6b7280', ha='center',
                     va='top', zorder=3, linespacing=1.5)
        if i < len(stages) - 1:
            ax_pipe.annotate('',
                xy=(x + box_w + gap, 0.1 + box_h / 2),
                xytext=(x + box_w, 0.1 + box_h / 2),
                arrowprops=dict(arrowstyle='->', color='#2d3748', lw=1.2),
                zorder=3)

    ax_pipe.set_xlim(0, 14)
    ax_pipe.set_ylim(0, 1.8)
    fig_pipe.tight_layout(pad=0)
    st.pyplot(fig_pipe)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # ── SECTION 2: DBSCAN CLUSTER VISUALISATION ──────────────
    st.markdown('<p class="section-label">DBSCAN clustering — how 115,350 violations became 951 hotspot clusters</p>',
                unsafe_allow_html=True)

    col_db1, col_db2 = st.columns(2)

    with col_db1:
        fig_cs, ax_cs = plt.subplots(figsize=(6, 3.5))
        fig_cs.patch.set_facecolor('#0d1117')
        ax_cs.set_facecolor('#111720')
        counts = pcis['violation_count']
        ax_cs.hist(counts, bins=50, color='#e07b39',
                   edgecolor='#0d1117', linewidth=0.4, alpha=0.85)
        ax_cs.axvline(counts.median(), color='#38a89d', linestyle='--',
                      linewidth=1.2, label=f'Median: {counts.median():.0f}')
        ax_cs.axvline(counts.mean(), color='#d4a843', linestyle='--',
                      linewidth=1.2, label=f'Mean: {counts.mean():.0f}')
        ax_cs.set_xlabel('Violations per cluster', color='#4a5568', fontsize=9)
        ax_cs.set_ylabel('Number of clusters',     color='#4a5568', fontsize=9)
        ax_cs.tick_params(colors='#4a5568', labelsize=8)
        ax_cs.spines[['top','right','bottom','left']].set_color('#1e2530')
        ax_cs.legend(facecolor='#111720', labelcolor='#8b95a1',
                     edgecolor='#1e2530', fontsize=8)
        ax_cs.set_title('Cluster Size Distribution',
                        color='#8b95a1', fontsize=9, pad=8)
        fig_cs.tight_layout()
        st.pyplot(fig_cs)

    with col_db2:
        fig_top, ax_top = plt.subplots(figsize=(6, 3.5))
        fig_top.patch.set_facecolor('#0d1117')
        ax_top.set_facecolor('#111720')
        top15      = pcis.nsmallest(15, 'pcis_rank')[['road_name','violation_count','severity_tier']]
        bar_colors = [TIER_COLORS.get(t, '#888') for t in top15['severity_tier']]
        ax_top.barh(
            [f"#{i+1} {n[:22]}" for i, n in enumerate(top15['road_name'])],
            top15['violation_count'],
            color=bar_colors, edgecolor='#0d1117', linewidth=0.4
        )
        ax_top.invert_yaxis()
        ax_top.set_xlabel('Violation count', color='#4a5568', fontsize=9)
        ax_top.tick_params(colors='#4a5568', labelsize=7.5)
        ax_top.spines[['top','right','bottom','left']].set_color('#1e2530')
        ax_top.set_title('Top 15 Hotspots by Violation Count',
                         color='#8b95a1', fontsize=9, pad=8)
        fig_top.tight_layout()
        st.pyplot(fig_top)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # ── SECTION 3: PCIS FORMULA BREAKDOWN ────────────────────
    st.markdown('<p class="section-label">PCIS formula breakdown — select a hotspot to inspect its score components</p>',
                unsafe_allow_html=True)

    top50_options = pcis.nsmallest(50, 'pcis_rank')['road_name'].tolist()
    selected_road = st.selectbox("Select hotspot", top50_options,
                                 label_visibility="collapsed")
    sel = pcis[pcis['road_name'] == selected_road].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    components = [
        (c1, "Violation Frequency",  f"{sel['violation_freq']:.4f}",
         "Log-normalized count\nacross all 951 clusters", '#e07b39'),
        (c2, "Road Capacity Factor", f"{sel['road_capacity_factor']:.3f}",
         f"Highway: {sel['highway_type']}\n0.15 (service) → 0.95 (trunk)", '#d4a843'),
        (c3, "Speed Degradation",    f"{sel['speed_degradation']:.4f}",
         "60% density proxy\n+ 40% road capacity", '#e05252'),
        (c4, "Peak Hour Weight",     f"{sel['peak_hour_weight']:.4f}",
         f"Peak ratio: {sel['peak_ratio']:.1%}\n1.0 off-peak → 2.0 all-peak", '#38a89d'),
        (c5, "PCIS Score",           f"{sel['PCIS_score']:.1f}",
         f"Tier: {sel['severity_tier']}\nRank #{int(sel['pcis_rank'])} of 951", '#6b7280'),
    ]
    for col, label, value, desc, color in components:
        with col:
            st.markdown(
                f'<div class="kpi-card" style="border-left:3px solid {color}">'
                f'<div class="kpi-label">{label}</div>'
                f'<div style="font-family:JetBrains Mono,monospace;font-size:22px;'
                f'font-weight:500;color:{color};line-height:1;margin-bottom:8px">{value}</div>'
                f'<div style="font-size:10px;color:#4a5568;line-height:1.5;'
                f'white-space:pre-line">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # Formula row
    fig_form, ax_form = plt.subplots(figsize=(12, 1.6))
    fig_form.patch.set_facecolor('#0d1117')
    ax_form.set_facecolor('#0d1117')
    ax_form.axis('off')

    items = [
        (f"{sel['violation_freq']:.4f}",       "Violation\nFrequency",  '#e07b39'),
        ('×', '',                                                         '#4a5568'),
        (f"{sel['road_capacity_factor']:.3f}", "Road\nCapacity",         '#d4a843'),
        ('×', '',                                                         '#4a5568'),
        (f"{sel['speed_degradation']:.4f}",    "Speed\nDegradation",     '#e05252'),
        ('×', '',                                                         '#4a5568'),
        (f"{sel['peak_hour_weight']:.4f}",     "Peak Hour\nWeight",      '#38a89d'),
        ('=', '',                                                         '#4a5568'),
        (f"{sel['PCIS_score']:.1f} / 100",     f"PCIS Score\n{sel['severity_tier']}", '#e2e8f0'),
    ]
    x_pos = 0.02
    for val, label, color in items:
        is_op = val in ('×', '=')
        ax_form.text(x_pos, 0.65, val,
                     fontsize=13 if is_op else 16, color=color,
                     fontfamily='monospace',
                     fontweight='700' if not is_op else '400', va='center')
        if label:
            ax_form.text(x_pos, 0.18, label,
                         fontsize=7, color='#4a5568', va='center', linespacing=1.4)
        x_pos += 0.06 if is_op else 0.13

    ax_form.set_xlim(0, 1)
    ax_form.set_ylim(0, 1)
    fig_form.tight_layout(pad=0)
    st.pyplot(fig_form)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # ── SECTION 4: MODEL PERFORMANCE ─────────────────────────
    st.markdown('<p class="section-label">Predictive model performance — XGBoost severity classifier</p>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value, color in [
        (m1, "Algorithm",     "XGBoost",      '#e07b39'),
        (m2, "R² Score",      "0.8644",        '#38a89d'),
        (m3, "MAE",           "5.056 pts",     '#d4a843'),
        (m4, "Training Size", "~92K records",  '#6b7280'),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-card" style="border-left:3px solid {color}">'
                f'<div class="kpi-label">{label}</div>'
                f'<div style="font-family:JetBrains Mono,monospace;font-size:18px;'
                f'font-weight:500;color:{color};line-height:1.2">{value}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">XGBoost feature importance</p>',
                unsafe_allow_html=True)

    if os.path.exists('stage5_xgb_feature_importance.png'):
        st.image('stage5_xgb_feature_importance.png', use_column_width=True)
    else:
        features    = ['latitude','longitude','hour','vehicle_type_enc',
                       'day_of_week','month','is_peak','is_weekend']
        importances = [0.31, 0.28, 0.17, 0.10, 0.07, 0.04, 0.02, 0.01]
        fig_fi, ax_fi = plt.subplots(figsize=(10, 3.5))
        fig_fi.patch.set_facecolor('#0d1117')
        ax_fi.set_facecolor('#111720')
        colors_fi = ['#e07b39' if v > 0.1 else '#1e3040' for v in importances]
        ax_fi.barh(features, importances, color=colors_fi,
                   edgecolor='#0d1117', linewidth=0.4)
        ax_fi.invert_yaxis()
        ax_fi.set_xlabel('Importance score', color='#4a5568', fontsize=9)
        ax_fi.tick_params(colors='#4a5568', labelsize=9)
        ax_fi.spines[['top','right','bottom','left']].set_color('#1e2530')
        fig_fi.tight_layout()
        st.pyplot(fig_fi)

    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # ── SECTION 5: PROPHET FORECAST ──────────────────────────
    st.markdown('<p class="section-label">Prophet 30-day forecast — top 20 hotspots</p>',
                unsafe_allow_html=True)

    if os.path.exists('stage5_prophet_forecasts.png'):
        st.image('stage5_prophet_forecasts.png', use_column_width=True)

    st.markdown(
        '<div style="margin-top:16px;padding:16px;background:#111720;'
        'border:1px solid #1e2530;border-radius:6px">'
        '<p style="font-size:10px;font-weight:700;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#6b7280;margin:0 0 10px 0">How Prophet works in PCIS</p>'
        '<p style="font-size:13px;color:#8b95a1;line-height:1.7;margin:0">'
        'Prophet decomposes each hotspot\'s daily violation count into trend, weekly seasonality '
        'and monthly seasonality components. It learns that violations spike on specific days '
        'and hours, then projects that pattern 30 days forward. '
        'This enables BTP to schedule patrol units proactively — before violations occur — '
        'rather than reacting after congestion has already formed.'
        '</p></div>',
        unsafe_allow_html=True
    )