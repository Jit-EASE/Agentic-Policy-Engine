# app.py

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---- Corrected imports for Option B folder structure ----
from core.config import (
    APP_TITLE,
    APP_SUBTITLE,
    SPECTRE_PRIMARY,
    SPECTRE_BG_GRADIENT,
)

from core.data_ingestion import load_csv_file
from core.schema_mapper import suggest_schema, build_canonical_long
from core.preprocess import coerce_time, filter_series, basic_clean
from core.logging_utils import log_error

from engines.engine_orchestrator import (
    run_queue_block,
    run_kalman_block,
    run_wavelet_block,
    run_mpc_block,
    run_routing_block,
)

from interfaces.ai_explainer import explain_block


# ---------------------- Spectre Holographic CSS ----------------------


def inject_spectre_css():
    st.markdown(
        f"""
        <style>
        html, body, [class*="stApp"] {{
            background: {SPECTRE_BG_GRADIENT};
            color: #e5ecff;
            font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
            font-size: 13px;
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }}
        h1, h2, h3, h4 {{
            font-weight: 600;
            letter-spacing: 0.04em;
        }}
        .spectre-card {{
            background: rgba(5, 10, 25, 0.72);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            border: 1px solid rgba(138, 180, 248, 0.32);
            box-shadow: 0 0 25px rgba(0, 200, 255, 0.16);
            backdrop-filter: blur(16px);
            transition: transform 160ms ease-out, box-shadow 160ms ease-out, border-color 160ms ease-out;
        }}
        .spectre-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 0 32px rgba(0, 220, 255, 0.32);
            border-color: rgba(138, 180, 248, 0.9);
        }}
        .spectre-pill {{
            display: inline-flex;
            padding: 2px 10px;
            border-radius: 999px;
            border: 1px solid rgba(138, 180, 248, 0.5);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            background: linear-gradient(90deg, rgba(138,180,248,0.18), rgba(3,218,197,0.12));
        }}
        .spectre-agent-box {{
            background: radial-gradient(circle at top left, rgba(4,14,40,0.95), rgba(1,3,10,0.98));
            border-radius: 18px;
            padding: 0.8rem 1rem;
            border: 1px solid rgba(3, 218, 197, 0.4);
            box-shadow: 0 0 20px rgba(3, 218, 197, 0.22);
            font-size: 12px;
        }}
        .spectre-small-label {{
            font-size: 11px;
            opacity: 0.75;
        }}
        .stTabs [data-baseweb="tab-list"] button {{
            font-size: 11px;
            padding-top: 4px;
            padding-bottom: 4px;
        }}
        .stSelectbox label, .stTextInput label, .stNumberInput label {{
            font-size: 11px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def spectre_title():
    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;gap:0.1rem;margin-bottom:0.7rem;">
          <div class="spectre-pill">Agentic Policy Engine</div>
          <div style="font-size:1.1rem;font-weight:650;margin-top:0.25rem;">
            {APP_TITLE}
          </div>
          <div style="font-size:0.9rem;color:rgba(197,208,255,0.75);margin-top:0.1rem;">
            {APP_SUBTITLE}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------- Utility: Plotly helpers ----------------------


def spectre_line_chart(x, y_dict: dict, title: str):
    fig = go.Figure()
    for name, series in y_dict.items():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=series,
                mode="lines",
                name=name,
                hovertemplate="%{y:.3f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        title=title,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#e5ecff"),
        margin=dict(l=30, r=20, t=35, b=25),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(120,130,160,0.3)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------- Main App ----------------------


def main():
    st.set_page_config(
        page_title="Agentic Policy Engine",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_spectre_css()

    with st.sidebar:
        st.markdown(f"### 🛰️ Spectre Control")
        st.caption("Upload any CSV / Excel and map columns to initialise the engines.")
        data_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])
        scenario_name = st.text_input("Scenario name", value="Irish Agri-Food Scenario")

    spectre_title()

    if data_file is None:
        st.info("Upload a dataset in the sidebar to unlock all tabs.")
        return

    # Load
    try:
        loaded = load_csv_file(data_file)
        df_raw = loaded["df"]
        meta = loaded["meta"]
    except Exception as e:
        log_error("Data ingestion", e)
        st.error(f"Could not read file: {e}")
        return

    with st.expander("📊 Data profile (Spectre Scanner)", expanded=False):
        st.write(f"File: `{meta['file_name']}` – {meta['n_rows']} rows × {meta['n_cols']} cols")
        st.dataframe(df_raw.head(25), use_container_width=True, height=220)

    # Schema suggestion + mapping UI
    suggested = suggest_schema(df_raw)
    st.markdown("#### Schema Mapper")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        time_col = st.selectbox("Time column", [None] + list(df_raw.columns), index=(list(df_raw.columns).index(suggested["time"]) + 1) if suggested["time"] in df_raw.columns else 0)
    with c2:
        region_col = st.selectbox("Region column", [None] + list(df_raw.columns), index=(list(df_raw.columns).index(suggested["region"]) + 1) if suggested["region"] in df_raw.columns else 0)
    with c3:
        metric_col = st.selectbox("Metric column", [None] + list(df_raw.columns), index=(list(df_raw.columns).index(suggested["metric"]) + 1) if suggested["metric"] in df_raw.columns else 0)
    with c4:
        value_col = st.selectbox("Value column", [None] + list(df_raw.columns), index=(list(df_raw.columns).index(suggested["value"]) + 1) if suggested["value"] in df_raw.columns else 0)

    try:
        canonical = build_canonical_long(df_raw, time_col, region_col, metric_col, value_col)
        canonical = coerce_time(canonical)
        canonical = basic_clean(canonical)
    except Exception as e:
        log_error("Schema mapping", e)
        st.error(f"Schema mapping failed: {e}")
        return

    # Prepare dropdowns
    regions = ["All"] + sorted(canonical["region"].dropna().unique().tolist())
    metrics = ["All"] + sorted(canonical["metric"].dropna().unique().tolist())

    tab_queue, tab_kalman, tab_mpc, tab_wavelet, tab_routing = st.tabs(
        ["Queueing ⏳", "Kalman Forecast 🔍", "MPC Control 🎛️", "Wavelets 🌊", "Routing 🧭"]
    )

    # ---------------- TAB 1: Queueing ----------------
    with tab_queue:
        st.markdown('<div class="spectre-card">', unsafe_allow_html=True)
        st.markdown("##### Queueing Theory – Policy Bottleneck Scanner")

        left, right = st.columns([2, 1])

        with left:
            st.caption("Configure an M/M/c queue system to approximate policy or processing bottlenecks.")

            arrival_rate = st.number_input("Arrival rate λ (e.g., applications per day)", min_value=0.0001, value=10.0)
            service_rate = st.number_input("Service rate μ per server (e.g., processed per day)", min_value=0.0001, value=5.0)
            servers = st.number_input("Number of servers c (capacity units)", min_value=1, step=1, value=2)

            if st.button("Run Queue Analysis", key="queue_run"):
                try:
                    qres = run_queue_block(arrival_rate, service_rate, servers)
                    st.success("Queue analysis completed.")
                    st.json(qres, expanded=False)

                    fig = spectre_line_chart(
                        x=[0, 1],
                        y_dict={"Utilisation ρ": [qres["utilisation_rho"], qres["utilisation_rho"]]},
                        title="Utilisation Level (ρ)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    queue_context = (
                        f"Arrival rate λ={arrival_rate}, service rate μ={service_rate}, servers={servers}, "
                        f"utilisation ρ={qres['utilisation_rho']:.3f}, stable={qres['stable']}. "
                        f"Expected waiting time Wq={qres['Wq_expected_wait_time']} and queue length Lq={qres['Lq_expected_queue_length']}."
                    )
                except Exception as e:
                    log_error("Queue engine", e)
                    st.error(f"Queue engine failed: {e}")
                    queue_context = "Queue engine failed."
            else:
                queue_context = "No queue scenario executed yet."

        with right:
            st.markdown("###### Spectre Agent – Queue Insight")
            agent_text = explain_block("Queueing", queue_context)
            st.markdown(f'<div class="spectre-agent-box">{agent_text}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TAB 2: Kalman ----------------
    with tab_kalman:
        st.markdown('<div class="spectre-card">', unsafe_allow_html=True)
        st.markdown("##### Kalman Smoothing – Real-Time Forecast Layer")

        left, right = st.columns([2, 1])

        with left:
            col1, col2 = st.columns(2)
            with col1:
                region_sel = st.selectbox("Region filter", regions, key="kalman_region")
            with col2:
                metric_sel = st.selectbox("Metric filter", metrics, key="kalman_metric")

            filtered = filter_series(canonical, region_sel, metric_sel)
            if filtered.empty:
                st.warning("No data after filtering. Adjust filters.")
                kalman_context = "No series available."
            else:
                st.caption("Selected series preview")
                st.dataframe(filtered.head(20), use_container_width=True, height=220)

                process_var = st.slider("Process variance (Q)", 0.1, 10.0, 1.0, 0.1)
                meas_var = st.slider("Measurement variance (R)", 0.1, 10.0, 4.0, 0.1)

                if st.button("Run Kalman Smoother", key="kalman_run"):
                    try:
                        series = filtered.set_index("time")["value"]
                        kres = run_kalman_block(series, process_var, meas_var)

                        fig = spectre_line_chart(
                            x=series.index,
                            y_dict={
                                "Observed": series,
                                "Filtered": kres["filtered"],
                                "Smoothed": kres["smoothed"],
                            },
                            title="Kalman Filter & Smoother",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        kalman_context = (
                            f"Region={region_sel}, metric={metric_sel}, "
                            f"process_var={process_var}, meas_var={meas_var}. "
                            f"Series length={len(series)}. Filter and smoother applied."
                        )
                    except Exception as e:
                        log_error("Kalman engine", e)
                        st.error(f"Kalman engine failed: {e}")
                        kalman_context = "Kalman engine failed."
                else:
                    kalman_context = "No Kalman run yet."

        with right:
            st.markdown("###### Spectre Agent – Forecast Insight")
            agent_text = explain_block("Kalman Forecast", kalman_context)
            st.markdown(f'<div class="spectre-agent-box">{agent_text}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TAB 3: MPC ----------------
    with tab_mpc:
        st.markdown('<div class="spectre-card">', unsafe_allow_html=True)
        st.markdown("##### MPC Trajectory – Emissions / Nitrates Control")

        left, right = st.columns([2, 1])

        with left:
            col1, col2 = st.columns(2)
            with col1:
                region_sel_mpc = st.selectbox("Region filter", regions, key="mpc_region")
            with col2:
                metric_sel_mpc = st.selectbox("Metric (emissions/nitrates proxy)", metrics, key="mpc_metric")

            filtered_mpc = filter_series(canonical, region_sel_mpc, metric_sel_mpc)
            if filtered_mpc.empty:
                st.warning("No data for MPC after filtering.")
                mpc_context = "No MPC run."
            else:
                series_mpc = filtered_mpc.set_index("time")["value"].sort_index()
                current_val = float(series_mpc.iloc[-1])
                st.caption(f"Current value (latest observation): {current_val:.3f}")

                target_val = st.number_input("Target value (e.g., emissions/nitrates cap)", value=max(current_val * 0.7, 0.0))
                horizon = st.slider("Horizon steps (years or periods)", 3, 20, 7)
                max_delta = st.number_input("Max absolute change per step (optional)", value=float(0.0))

                if st.button("Generate MPC-style Trajectory", key="mpc_run"):
                    try:
                        max_delta_use = None if max_delta <= 0 else max_delta
                        traj = run_mpc_block(current_val, target_val, horizon, max_delta_use)

                        fig = spectre_line_chart(
                            x=traj.index,
                            y_dict={"Trajectory": traj},
                            title="MPC-style Adjustment Path",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        mpc_context = (
                            f"MPC trajectory from {current_val:.3f} to {target_val:.3f} "
                            f"over horizon={horizon}, max_delta={max_delta_use}."
                        )
                    except Exception as e:
                        log_error("MPC engine", e)
                        st.error(f"MPC engine failed: {e}")
                        mpc_context = "MPC engine failed."
                else:
                    mpc_context = "No MPC trajectory generated yet."

        with right:
            st.markdown("###### Spectre Agent – Control Insight")
            agent_text = explain_block("MPC Control", mpc_context)
            st.markdown(f'<div class="spectre-agent-box">{agent_text}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TAB 4: Wavelets ----------------
    with tab_wavelet:
        st.markdown('<div class="spectre-card">', unsafe_allow_html=True)
        st.markdown("##### Wavelet Decomposition – Trend / Shocks Split")

        left, right = st.columns([2, 1])

        with left:
            col1, col2 = st.columns(2)
            with col1:
                region_sel_w = st.selectbox("Region filter", regions, key="wav_region")
            with col2:
                metric_sel_w = st.selectbox("Metric filter", metrics, key="wav_metric")

            filtered_w = filter_series(canonical, region_sel_w, metric_sel_w)
            if filtered_w.empty:
                st.warning("No data for wavelet decomposition.")
                wav_context = "No wavelet run."
            else:
                series_w = filtered_w.set_index("time")["value"].sort_index()
                if st.button("Run Wavelet Decomposition", key="wave_run"):
                    try:
                        wres = run_wavelet_block(series_w)
                        fig = spectre_line_chart(
                            x=series_w.index,
                            y_dict={
                                "Original": series_w,
                                "Trend": wres["trend"],
                                "Shocks": wres["shocks"],
                            },
                            title="Wavelet Trend / Shock Decomposition",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        wav_context = (
                            f"Wavelet decomposition for region={region_sel_w}, metric={metric_sel_w}, "
                            f"series length={len(series_w)}. Trend and shocks isolated."
                        )
                    except Exception as e:
                        log_error("Wavelet engine", e)
                        st.error(f"Wavelet engine failed: {e}")
                        wav_context = "Wavelet engine failed."
                else:
                    wav_context = "No wavelet run yet."

        with right:
            st.markdown("###### Spectre Agent – Signal Insight")
            agent_text = explain_block("Wavelets", wav_context)
            st.markdown(f'<div class="spectre-agent-box">{agent_text}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TAB 5: Routing ----------------
    with tab_routing:
        st.markdown('<div class="spectre-card">', unsafe_allow_html=True)
        st.markdown("##### Routing Optimiser – Logistics Simulation")

        left, right = st.columns([2, 1])

        with left:
            st.caption("Provide a small table of locations (id, lat, lon) inside your dataset, or paste a subset.")

            # Try to detect lat/lon
            candidate_lat = [c for c in df_raw.columns if "lat" in c.lower()]
            candidate_lon = [c for c in df_raw.columns if "lon" in c.lower() or "long" in c.lower()]
            candidate_id = [c for c in df_raw.columns if c.lower() in ("id", "node", "name", "location")]

            if candidate_id and candidate_lat and candidate_lon:
                id_col = st.selectbox("ID column", candidate_id)
                lat_col = st.selectbox("Latitude column", candidate_lat)
                lon_col = st.selectbox("Longitude column", candidate_lon)

                loc_df = df_raw[[id_col, lat_col, lon_col]].dropna().rename(
                    columns={id_col: "id", lat_col: "lat", lon_col: "lon"}
                )
                st.dataframe(loc_df.head(15), use_container_width=True, height=200)

                depot_id = st.selectbox("Select depot / hub ID", loc_df["id"].unique().tolist())

                if st.button("Compute Shortest Routes", key="routing_run"):
                    try:
                        routes = run_routing_block(loc_df, depot_id)
                        st.success("Routing computed.")

                        for dest, path, dist in routes:
                            st.markdown(
                                f"<div class='spectre-small-label'>Route to <b>{dest}</b>: "
                                f"{' → '.join(path)} — {dist:.1f} km</div>",
                                unsafe_allow_html=True,
                            )

                        routing_context = (
                            f"Routing from depot={depot_id}, {len(routes)} destination routes computed based on haversine distances."
                        )
                    except Exception as e:
                        log_error("Routing engine", e)
                        st.error(f"Routing engine failed: {e}")
                        routing_context = "Routing engine failed."
                else:
                    routing_context = "Routing not executed yet."
            else:
                st.warning("Could not auto-detect id/lat/lon columns. Add them to your dataset to enable routing.")
                routing_context = "No routing possible without id/lat/lon."

        with right:
            st.markdown("###### Spectre Agent – Logistics Insight")
            agent_text = explain_block("Routing", routing_context)
            st.markdown(f'<div class="spectre-agent-box">{agent_text}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
