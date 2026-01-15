"""
UI - Folium Map Renderer
=============================================
Description: Renders interactive forecast maps using Folium with a ForeFlight-
             inspired aesthetic (Sanctum Glassmorphism). Features dual-layer
             visualization (Live Radar + 15-min Forecast), airport markers with
             demand metrics, and a responsive HUD panel.
Author: RT Forecast Team
Version: 3.8.0

Visual Design:
    - Dark mode with NEXRAD-style gradient (Green -> Yellow -> Red -> Magenta)
    - Glass-effect panels with backdrop blur
    - Toggle between Live Radar and Flow Forecast layers
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap

from .config import CORE_BBOX, GRID_RES, AIRPORTS, NEXRAD_GRADIENT, HTML_OUT
from .preproc import filter_to_core

log = logging.getLogger('rt_forecast.ui')

# === CONSTANTS ===
ADVECTION_HOURS = 0.25
NM_PER_DEGREE = 60.0
MIN_CONGESTION_THRESHOLD = 0.001
MAX_DISPLACEMENT_DEG = 0.6
VISUAL_ADVECTION_BOOST = 1.2
MARKER_Z_INDEX = 2500


def render_forecast(
    predictions: Dict[str, pd.DataFrame],
    states: List[Dict],
    metrics: Dict[str, Any],
    airport_scores: Dict[str, Dict]
) -> bool:
    """Render ForeFlight-style interactive forecast map."""
    try:
        sector_bounds = [[36.5, -124.0], [38.5, -120.5]]
        
        m = folium.Map(
            location=[37.5, -122.1],
            zoom_start=9,
            min_zoom=9,
            max_zoom=12,
            max_bounds=True,
            min_lat=sector_bounds[0][0],
            max_lat=sector_bounds[1][0],
            min_lon=sector_bounds[0][1],
            max_lon=sector_bounds[1][1],
            tiles='CartoDB dark_matter',
            prefer_canvas=True,
            zoom_control=False
        )

        # Layer config: (key, column, name, show_default, radius, blur, max_zoom)
        # Forecast uses smaller radius to prevent overlap blowup on zoom
        layer_config = [
            ('current', 'current_congestion', 'Live Radar', True, 40, 35, 10),
            ('15min', 'predicted_15min', 'Flow Forecast', False, 45, 40, 10)
        ]

        # Track layer JS variable names for toggle
        layer_refs = {}

        for key, pred_col, layer_name, show_default, rad, blr, max_z in layer_config:
            df = predictions.get(key, pd.DataFrame())
            if df.empty or pred_col not in df.columns:
                log.warning(f"Layer '{layer_name}': missing data or column '{pred_col}'")
                continue

            visible_df = filter_to_core(df)
            use_efc = (key == '15min')
            threshold = MIN_CONGESTION_THRESHOLD * 0.5 if use_efc else MIN_CONGESTION_THRESHOLD

            heat_data = _build_heat_data(visible_df, pred_col, apply_advection=use_efc, min_threshold=threshold)

            fg = folium.FeatureGroup(name=layer_name, show=show_default)

            if heat_data:
                HeatMap(
                    heat_data,
                    min_opacity=0.15,
                    max_opacity=0.80 if not use_efc else 0.70,  # Forecast slightly lower max
                    radius=rad,
                    blur=blr,
                    max_zoom=max_z,
                    gradient=NEXRAD_GRADIENT
                ).add_to(fg)
            
            fg.add_to(m)
            # Capture the JS variable name Folium generates
            layer_refs[layer_name] = fg.get_name()
            log.debug(f"Layer '{layer_name}' -> JS var: {fg.get_name()}")

        _inject_theme(m, layer_refs)
        _add_airport_markers(m, predictions, airport_scores)
        _add_hud(m, metrics, airport_scores)

        m.save(str(HTML_OUT))
        log.info(f"Map rendered: {HTML_OUT}")
        return True

    except Exception as e:
        log.error(f"UI Failure: {e}", exc_info=True)
        return False


def _build_heat_data(df: pd.DataFrame, pred_col: str, apply_advection: bool = False, min_threshold: float = None) -> List[List[float]]:
    threshold = min_threshold if min_threshold is not None else MIN_CONGESTION_THRESHOLD
    df_clean = df[df[pred_col] >= threshold].copy()

    if df_clean.empty:
        if not df.empty:
            df_clean = df.copy()
        else:
            return []

    lats = df_clean['lat_bin'].values + GRID_RES / 2
    lons = df_clean['lon_bin'].values + GRID_RES / 2
    
    # Unified rendering: both layers use same visual mapping now that inference is normalized
    # The max_zoom=10 fix prevents zoom-in blowup, so we can use consistent scaling
    vals = np.clip(df_clean[pred_col].values * 0.6, 0.01, 1.0)

    if apply_advection:
        shift = (ADVECTION_HOURS / NM_PER_DEGREE) * VISUAL_ADVECTION_BOOST
        lats += np.clip(df_clean['avg_v'].fillna(0).values * shift, -MAX_DISPLACEMENT_DEG, MAX_DISPLACEMENT_DEG)
        lons += np.clip(df_clean['avg_u'].fillna(0).values * shift, -MAX_DISPLACEMENT_DEG, MAX_DISPLACEMENT_DEG)

    return np.column_stack([lats, lons, vals]).tolist()


def _inject_theme(m: folium.Map, layer_refs: Dict[str, str]):
    """ForeFlight aesthetics with Flight Ops panel."""
    
    favicon = """
    <link rel="icon" href="data:image/svg+xml,
    <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>
        <text y='0.9em' font-size='90'>✈️</text>
    </svg>">
    """

    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@500&display=swap');
    :root { --glass: rgba(15, 15, 15, 0.85); --blue: #00BFFF; --cyan: #00FFFF; }
    .leaflet-container { background: #050505 !important; }

    .sanctum-panel { position: fixed; top: 20px; right: 20px; z-index: 10000; background: var(--glass); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; color: white; font-family: 'Inter', sans-serif; width: 180px; }
    .toggle-row { display: flex; align-items: center; margin: 8px 0; cursor: pointer; padding: 8px; border-radius: 6px; }
    .toggle-row input { margin-right: 12px; accent-color: var(--blue); }
    .toggle-row.active { color: var(--blue); font-weight: 600; }

    .leaflet-popup-content-wrapper { background: rgba(255, 255, 255, 0.98); color: #111; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .leaflet-popup-tip { background: rgba(255, 255, 255, 0.98); }

    .efb-popup { font-family: 'Inter', sans-serif; color: #111; min-width: 170px; }
    .efb-header { display: flex; justify-content: space-between; align-items: center; font-weight: 600; font-size: 14px; border-bottom: 1px solid #ddd; margin-bottom: 8px; padding-bottom: 4px; }
    .rank-badge { background: #333; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
    .rank-badge.rank-1 { background: #d32f2f; }
    .rank-badge.rank-2 { background: #f57c00; }
    .rank-badge.rank-3 { background: #fbc02d; color: #111; }

    .efb-stat { display: flex; justify-content: space-between; font-size: 12px; margin: 5px 0; }
    .efb-val { font-family: 'JetBrains Mono', monospace; color: #000; font-weight: 600; }

    .leaflet-interactive { cursor: pointer !important; }
    .airport-label { color: var(--cyan); font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; text-shadow: 1px 1px 2px #000; pointer-events: none !important; }
    .leaflet-heatmap-layer { pointer-events: none !important; }
    </style>
    """

    # Direct Folium variable references - no window iteration needed
    map_var = m.get_name()
    cur_var = layer_refs.get('Live Radar', 'null')
    fct_var = layer_refs.get('Flow Forecast', 'null')

    js = f"""
    <script>
    (function() {{
        var box = document.createElement('div'); 
        box.className = 'sanctum-panel';
        box.innerHTML = '<div style="font-size:10px; color:#888; letter-spacing:1px; margin-bottom:10px">FLIGHT DECK OPS</div>' +
            '<label class="toggle-row active" id="l-cur"><input type="radio" name="radar" value="cur" checked> LIVE RADAR</label>' +
            '<label class="toggle-row" id="l-fct"><input type="radio" name="radar" value="fct"> 15M FORECAST</label>';
        document.body.appendChild(box);

        function syncLayers(mode) {{
            // Direct references to Folium-generated JS variables (no window iteration!)
            var map = (typeof {map_var} !== 'undefined') ? {map_var} : null;
            var cur = (typeof {cur_var} !== 'undefined') ? {cur_var} : null;
            var fct = (typeof {fct_var} !== 'undefined') ? {fct_var} : null;

            if (!map) {{ console.warn('RT Forecast: Map not ready'); return; }}

            var lc = document.getElementById('l-cur');
            var lf = document.getElementById('l-fct');

            if (mode === 'cur') {{
                if (cur && !map.hasLayer(cur)) map.addLayer(cur);
                if (fct && map.hasLayer(fct)) map.removeLayer(fct);
                lc.classList.add('active'); 
                lf.classList.remove('active');
            }} else {{
                if (fct && !map.hasLayer(fct)) map.addLayer(fct);
                if (cur && map.hasLayer(cur)) map.removeLayer(cur);
                lf.classList.add('active'); 
                lc.classList.remove('active');
            }}
        }}

        // Initialize on load with small delay to ensure Folium vars exist
        window.addEventListener('load', function() {{ 
            setTimeout(function() {{ syncLayers('cur'); }}, 200); 
        }});

        box.querySelector('input[value="cur"]').onchange = function() {{ syncLayers('cur'); }};
        box.querySelector('input[value="fct"]').onchange = function() {{ syncLayers('fct'); }};
    }})();
    </script>
    """
    m.get_root().header.add_child(folium.Element(favicon))
    m.get_root().header.add_child(folium.Element(css))
    m.get_root().html.add_child(folium.Element(js))


def _add_airport_markers(m: folium.Map, predictions: Dict, airport_scores: Dict):
    """Add markers with rank badges and trend indicators."""
    # Use dedicated FeatureGroup for markers (ensures they stay on top and clickable)
    marker_group = folium.FeatureGroup(name='Airports', show=True)

    for icao, (lat, lon, term_rad, det_rad, name) in AIRPORTS.items():
        scores = airport_scores.get(icao, {'rank': 5, 'score_current': 0.0, 'demand': 0})
        rank = scores.get('rank', 5)
        cur_s = scores.get('score_current', 0.0)
        demand = scores.get('demand', 0)

        rank_class = f"rank-{rank}" if rank <= 3 else ""

        popup_html = f'''<div class="efb-popup">
            <div class="efb-header">
                <span>{icao} | {name}</span>
                <span class="rank-badge {rank_class}">#{rank}</span>
            </div>
            <div class="efb-stat"><span>SECTOR DEMAND</span><span class="efb-val">{demand} AC</span></div>
        </div>'''

        # Label (non-interactive)
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(icon_anchor=(-12, 9), html=f'<div class="airport-label">{icao}</div>'),
            z_index_offset=MARKER_Z_INDEX
        ).add_to(marker_group)

        # Dot marker with popup (interactive)
        fill_color = "#00FF00" if cur_s < 0.4 else "#FFFF00" if cur_s < 0.7 else "#FF0000"
        folium.CircleMarker(
            [lat, lon], radius=6, color="white", weight=1, fill=True,
            fill_color=fill_color, fill_opacity=1.0,
            popup=folium.Popup(popup_html, max_width=220),
            z_index_offset=MARKER_Z_INDEX
        ).add_to(marker_group)

    # Add marker group to map LAST to ensure it's on top
    marker_group.add_to(m)


def _add_hud(m: folium.Map, metrics: Dict, airport_scores: Dict):
    """Clean HUD with 5-minute refresh indicator."""
    ts = metrics['window_time'].strftime('%H:%M UTC')
    status = metrics['status']
    status_col = "#00FF00" if status == "LIVE" else "#00FFFF" if status == "REPLAY" else "#FFA500"

    # Find busiest airport by demand
    busiest_icao = "N/A"
    busiest_demand = 0
    for icao, data in airport_scores.items():
        demand = data.get('demand', 0)
        if demand > busiest_demand:
            busiest_demand = demand
            busiest_icao = icao

    html = f"""
    <div style="position:fixed; bottom:20px; left:20px; z-index:10000;
                background:rgba(10,10,10,0.85); backdrop-filter:blur(10px);
                border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:15px;
                color:white; font-family: 'JetBrains Mono', monospace; font-size:11px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5); min-width:210px;">
        <div style="color:#00BFFF; font-weight:600; font-size:13px; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px;">
            Airspace Congestion Tracker
        </div>
        <div style="display:grid; grid-template-columns: 80px 1fr; gap:8px; line-height:1.6;">
            <span style="color:#777;">CLOCK</span> <span>{ts}</span>
            <span style="color:#777;">STATUS</span> <span style="color:{status_col}; font-weight:600;">{status}</span>
            <span style="color:#777;">BUSIEST</span> <span>{busiest_icao} ({busiest_demand} AC)</span>
        </div>
        <div style="margin-top:12px; color:#555; font-size:9px; text-align:center; border-top:1px solid #222; padding-top:8px; letter-spacing:0.5px;">
            UPDATES EVERY 5 MINUTES
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def compute_airport_scores(predictions: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Calculate relative airport ranking using dual-radius system."""
    scores = {}
    current_df = predictions.get('current', pd.DataFrame())

    if current_df.empty:
        for icao in AIRPORTS:
            scores[icao] = {'score_current': 0.0, 'rank': 5, 'demand': 0}
        return scores

    raw_metrics = []

    for icao, (lat, lon, term_rad, det_rad, _) in AIRPORTS.items():
        dist = np.sqrt((current_df['lat_bin'] - lat)**2 + (current_df['lon_bin'] - lon)**2)

        inclusive_mask = dist <= det_rad
        demand = int(current_df.loc[inclusive_mask, 'aircraft_count'].sum()) if inclusive_mask.any() else 0

        terminal_mask = dist <= term_rad
        cur_score = current_df.loc[terminal_mask, 'current_congestion'].mean() if terminal_mask.any() and 'current_congestion' in current_df.columns else 0.0

        if inclusive_mask.any() and 'flow_pressure' in current_df.columns:
            weights = 1.0 / (dist[inclusive_mask] + 0.1)
            raw_fp = current_df.loc[inclusive_mask, 'flow_pressure'].fillna(0)
            weighted_fp = (raw_fp * weights).sum() / weights.sum() if weights.sum() > 0 else 0.0
        else:
            weighted_fp = 0.0

        composite_load = (demand / 10.0) + (weighted_fp if not np.isnan(weighted_fp) else 0.0)

        raw_metrics.append({
            'icao': icao,
            'load': composite_load,
            'demand': demand,
            'cur_score': cur_score if not np.isnan(cur_score) else 0.0
        })

    ranked = sorted(raw_metrics, key=lambda x: x['load'], reverse=True)

    for i, item in enumerate(ranked, 1):
        scores[item['icao']] = {
            'score_current': item['cur_score'],
            'rank': min(i, 5),
            'demand': item['demand']
        }

    return scores