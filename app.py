import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter

from data_fetch import fetch_seasons
from elo import run_elo, build_leaderboard, expected_woba, compute_park_factors

st.set_page_config(page_title="Baseball ELO", layout="wide")
st.title("Baseball ELO Ratings — Iteration 2: wOBA + Park Factors")


@st.cache_data(show_spinner=False)
def cached_fetch(years: tuple[int, ...]):
    return fetch_seasons(list(years))


@st.cache_data(show_spinner=False)
def cached_elo(years: tuple[int, ...], league_woba: float):
    df, _ = cached_fetch(years)
    return run_elo(df, league_woba=league_woba, years=years)


@st.cache_data(show_spinner=False)
def season_bundle(year: int):
    """Single-season ratings + names + league wOBA for the matchup predictor."""
    df_y, names_y = cached_fetch((year,))
    lw = round(float(df_y["woba_value"].mean()), 4)
    res = cached_elo((year,), lw)
    return df_y, names_y, lw, res


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    from datetime import date as _date
    current_year = _date.today().year
    all_years = list(range(current_year, 2014, -1))

    scope = st.radio(
        "Scope", ["Single Season", "Career"], index=0,
        help="Single Season: ratings reset to 1500 each year. Career: ratings carry across seasons.",
    )
    if scope == "Single Season":
        year = st.selectbox("Season", all_years, index=0)
        selected_years = (year,)
        scope_label = str(year)
    else:
        years_asc = sorted(all_years)
        c1, c2 = st.columns(2)
        start_year = c1.selectbox("From", years_asc, index=max(0, len(years_asc) - 3))
        end_year = c2.selectbox("To", all_years, index=0)
        if end_year < start_year:
            start_year, end_year = end_year, start_year
        selected_years = tuple(range(start_year, end_year + 1))
        scope_label = f"{start_year}–{end_year} career"
        st.caption("Uncached seasons download ~a few min each on first use.")

    min_pa = st.slider("Min PA for leaderboard", 10, 1000, 100, 10)
    sort_by = st.radio("Sort leaderboard by", ["End ELO", "Avg ELO", "Peak ELO", "Worst ELO", "Range"],
                       help="End ELO: current rating. Peak ELO: hottest point. Worst ELO: lowest point. Avg ELO: sustained value. Range: Peak minus Worst (streakiness).")
    st.caption("Iteration 1 — on-base vs out only")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {scope_label} Statcast data (first run: a few minutes per season)…"):
    df, names = cached_fetch(selected_years)

league_woba = round(float(df["woba_value"].mean()), 4)

# ── Team lookup: most recent team for each batter and pitcher ─────────────────
# inning_topbot: 'Top' = away bats, 'Bot' = home bats
df_sorted = df.sort_values(["game_date", "game_pk", "at_bat_number"])
df_sorted["batter_team"] = np.where(df_sorted["inning_topbot"] == "Bot",
                                    df_sorted["home_team"], df_sorted["away_team"])
df_sorted["pitcher_team"] = np.where(df_sorted["inning_topbot"] == "Bot",
                                     df_sorted["away_team"], df_sorted["home_team"])
batter_teams: dict[int, str] = df_sorted.groupby("batter")["batter_team"].last().to_dict()
pitcher_teams: dict[int, str] = df_sorted.groupby("pitcher")["pitcher_team"].last().to_dict()
all_teams = sorted({t for t in set(batter_teams.values()) | set(pitcher_teams.values()) if t})

st.sidebar.markdown("---")
st.sidebar.subheader("Team Filter")
team_filter_raw = st.sidebar.multiselect("Filter leaderboards by team", all_teams,
                                          label_visibility="collapsed")
team_filter = set(team_filter_raw) if team_filter_raw else None

st.success(f"{scope_label} | {len(df):,} plate appearances | {len(names):,} players | League wOBA: {league_woba:.3f}")

# ── Run ELO ───────────────────────────────────────────────────────────────────
(batter_ratings, pitcher_ratings,
 batter_avg, pitcher_avg,
 batter_peak, pitcher_peak,
 batter_worst, pitcher_worst,
 batter_history, pitcher_history) = cached_elo(selected_years, league_woba)

# ── Build display names (append ID only if two players share a name) ──────────
# Built after run_elo so we can cover any IDs missing from the names lookup.
name_counts = Counter(names.values())
def _display(pid: int) -> str:
    name = names.get(pid, f"ID:{pid}")
    return f"{name} ({pid})" if name_counts.get(name, 1) > 1 else name

display_names: dict[int, str] = {
    pid: _display(pid)
    for pid in set(names) | set(batter_ratings) | set(pitcher_ratings)
}

batter_pa = df.groupby("batter").size()
pitcher_pa = df.groupby("pitcher").size()

leaderboard = build_leaderboard(batter_ratings, batter_avg, batter_peak, batter_worst, display_names, batter_pa, min_pa, sort_by, team_filter, batter_teams)
pitcher_board = build_leaderboard(pitcher_ratings, pitcher_avg, pitcher_peak, pitcher_worst, display_names, pitcher_pa, min_pa, sort_by, team_filter, pitcher_teams)

# ── Leaderboards ──────────────────────────────────────────────────────────────
st.subheader("Batters")
st.dataframe(leaderboard, use_container_width=True, hide_index=True, height=600)

st.subheader("Pitchers")
st.dataframe(pitcher_board, use_container_width=True, hide_index=True, height=600)

# ── Matchup predictor (cross-year) ────────────────────────────────────────────
st.subheader("Matchup Predictor")
st.caption(
    "Pick a season for each player, then the player. Ratings are league-relative "
    "(each season recenters to 1500), so cross-year matchups ask: how would these two "
    "relative skill levels fare in the chosen run environment? Avg ELO = sustained "
    "season level (recommended); End/Peak available too."
)

_metric_idx = {"Avg ELO": 2, "End ELO": 0, "Peak ELO": 4}


def _season_player_options(year: int, role: str, metric: str, min_pa_mp: int = 50):
    """Return (label->（pid, rating), names, df, league_woba) for one season+role."""
    df_y, names_y, lw, res = season_bundle(year)
    ratings = res[_metric_idx[metric] + (0 if role == "batter" else 1)]
    id_col = "batter" if role == "batter" else "pitcher"
    pa = df_y.groupby(id_col).size()
    counts = Counter(names_y.values())
    opts = {}
    for pid, rating in ratings.items():
        if pa.get(pid, 0) < min_pa_mp:
            continue
        nm = names_y.get(pid, f"ID:{pid}")
        label = f"{nm} ({pid})" if counts.get(nm, 1) > 1 else nm
        opts[label] = (pid, rating)
    return opts, names_y, df_y, lw


mp_metric = st.radio("Rating to use", ["Avg ELO", "End ELO", "Peak ELO"], horizontal=True, key="mp_metric")

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown("**Batter**")
    b_year = st.selectbox("Batter season", all_years, index=0, key="mp_b_year")
    b_opts, _, b_df, _ = _season_player_options(b_year, "batter", mp_metric)
    batter_pick = st.selectbox("Batter", [""] + sorted(b_opts), key="mp_batter")
with mc2:
    st.markdown("**Pitcher**")
    p_year = st.selectbox("Pitcher season", all_years, index=0, key="mp_p_year")
    p_opts, _, p_df, _ = _season_player_options(p_year, "pitcher", mp_metric)
    pitcher_pick = st.selectbox("Pitcher", [""] + sorted(p_opts), key="mp_pitcher")
with mc3:
    st.markdown("**Run environment**")
    env_year = st.selectbox("Season (sets league wOBA)", all_years, index=0, key="mp_env_year")
    _, _, env_lw, _ = season_bundle(env_year)
    st.metric(f"{env_year} league wOBA", f"{env_lw:.3f}")

if batter_pick and pitcher_pick:
    b_id, r_b = b_opts[batter_pick]
    p_id, r_p = p_opts[pitcher_pick]
    exp_woba = expected_woba(r_b, r_p, env_lw)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(f"Batter {mp_metric}", f"{r_b:.0f}", help=f"{batter_pick}, {b_year}")
    r2.metric(f"Pitcher {mp_metric}", f"{r_p:.0f}", help=f"{pitcher_pick}, {p_year}")
    r3.metric("Expected wOBA", f"{exp_woba:.3f}", help=f"In {env_year} run environment")
    r4.metric(f"vs {env_year} league", f"{exp_woba - env_lw:+.3f}")

    # Real head-to-head only exists if both came from the same season
    if b_year == p_year:
        h2h = b_df[(b_df["batter"] == b_id) & (b_df["pitcher"] == p_id)]
        if len(h2h) > 0:
            with st.expander(f"Real head-to-head, {b_year} ({len(h2h)} PA)"):
                h2h_display = h2h[["game_date", "events", "woba_value"]].copy()
                h2h_display["Result"] = h2h_display["events"].str.replace("_", " ").str.title()
                h2h_display = h2h_display[["game_date", "Result", "woba_value"]].rename(
                    columns={"game_date": "Date", "woba_value": "wOBA"})
                st.dataframe(h2h_display.sort_values("Date", ascending=False),
                             use_container_width=True, hide_index=True)
    else:
        st.caption(f"Hypothetical cross-era matchup ({b_year} batter vs {p_year} pitcher) — no real head-to-head.")

# ── Rating distribution ───────────────────────────────────────────────────────
st.subheader("Rating Distribution")
dist_weighted = st.checkbox(
    "Weight by plate appearances", value=True,
    help="On: each player counts proportional to PAs — both pools center at 1500 "
         "(the calibration). Off: raw player counts, which skew because low-PA "
         "batters and pitchers are distributed differently.",
)

b_vals = np.array([batter_ratings[p] for p in batter_ratings])
b_w = np.array([batter_pa.get(p, 0) for p in batter_ratings])
p_vals = np.array([pitcher_ratings[p] for p in pitcher_ratings])
p_w = np.array([pitcher_pa.get(p, 0) for p in pitcher_ratings])

lo = min(b_vals.min(), p_vals.min())
hi = max(b_vals.max(), p_vals.max())
bins = np.linspace(lo, hi, 61)
centers = (bins[:-1] + bins[1:]) / 2
weights_b = b_w if dist_weighted else None
weights_p = p_w if dist_weighted else None
b_hist, _ = np.histogram(b_vals, bins=bins, weights=weights_b)
p_hist, _ = np.histogram(p_vals, bins=bins, weights=weights_p)

fig = go.Figure()
fig.add_trace(go.Bar(x=centers, y=b_hist, name="Batters", opacity=0.6))
fig.add_trace(go.Bar(x=centers, y=p_hist, name="Pitchers", opacity=0.6))
fig.add_vline(x=1500, line_dash="dash", line_color="gray")
fig.update_layout(barmode="overlay", xaxis_title="ELO Rating",
                  yaxis_title="Plate appearances" if dist_weighted else "Players")
st.plotly_chart(fig, use_container_width=True)

# ── Player rating over time ───────────────────────────────────────────────────
st.subheader("ELO Rating Over Time")

# Display name → player id for the loaded scope (drives the time-series picker)
batter_label_to_id = {display_names[pid]: pid for pid in batter_ratings}
pitcher_label_to_id = {display_names[pid]: pid for pid in pitcher_ratings}

def build_compressed_chart(histories: dict[str, list], role_label: str,
                           names: dict[int, str]) -> go.Figure:
    """
    Date-based x-axis with off-days compressed.
    Active days (any selected player has a PA) get 1 unit each.
    Gaps between active days are compressed to 0.2 * actual_gap_days,
    so a 5-day break looks slightly wider than a 1-day gap but doesn't
    dominate. Flat lines carry a player's ELO across their own off-days.
    """
    from collections import defaultdict

    # Union of all active dates across all players
    all_active = sorted({
        date
        for history in histories.values()
        for _, date, *_ in history
    })
    if not all_active:
        return go.Figure()

    # Assign compressed x positions to every active date
    date_to_x: dict = {}
    x = 0.0
    prev = None
    for date in all_active:
        if prev is not None:
            gap = (date - prev).days
            # Compress off-days; cap total gap width at 5 units so a long
            # break (injury, or the offseason in career mode) can't dominate.
            x += min(1.0 + 0.2 * (gap - 1), 5.0)
        date_to_x[date] = x
        prev = date

    # Month-start tick marks. Multi-year spans label the year too.
    multi_year = all_active[0].year != all_active[-1].year
    tick_fmt = "%b %Y" if multi_year else "%b %d"
    tickvals, ticktext = [], []
    seen_months: set = set()
    for date in all_active:
        mk = (date.year, date.month)
        if mk not in seen_months:
            seen_months.add(mk)
            tickvals.append(date_to_x[date])
            ticktext.append(date.strftime(tick_fmt))

    fig = go.Figure()

    for label, history in histories.items():
        by_date: dict = defaultdict(list)
        for pa_num, date, rating, event, opponent_id, delta, opp_elo in history:
            by_date[date].append((pa_num, rating, event, opponent_id, delta, opp_elo))

        xs, ys, hovers = [], [], []
        last_rating = None
        last_x = None

        for date in all_active:
            base_x = date_to_x[date]
            pas = by_date.get(date)

            if pas is None:
                # Player was off — extend flat line to this date's position
                if last_rating is not None:
                    xs.append(base_x)
                    ys.append(last_rating)
                    hovers.append(f"{date.strftime('%b %d')} | did not play")
                continue

            n = len(pas)
            # Flat carry-over to start of this game
            if last_rating is not None and last_x != base_x:
                xs.append(base_x)
                ys.append(last_rating)
                hovers.append(f"{date.strftime('%b %d')} | game start | ELO: {last_rating}")

            for i, (pa_num, rating, event, opponent_id, delta, opp_elo) in enumerate(pas):
                pa_x = base_x + (i / (n - 1) * 0.8 if n > 1 else 0.0)
                opponent = names.get(opponent_id, f"ID:{opponent_id}")
                result = event.replace("_", " ").title()
                opp_label = "vs P" if role_label == "PA" else "vs B"
                sign = "+" if delta >= 0 else ""
                hovers.append(
                    f"{date.strftime('%b %d')} | {role_label} #{pa_num}<br>"
                    f"{opp_label}: {opponent} (ELO {opp_elo})<br>"
                    f"Result: {result}<br>"
                    f"ELO: {rating} ({sign}{delta})"
                )
                xs.append(pa_x)
                ys.append(rating)

            last_rating = pas[-1][1]
            last_x = base_x

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name=label, text=hovers, hoverinfo="text+name",
        ))

    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=-45)
    fig.update_layout(yaxis_title="ELO Rating", xaxis_title="Date (off-days compressed)")
    return fig


tab_b, tab_p = st.tabs(["Batters", "Pitchers"])

with tab_b:
    options = sorted(batter_label_to_id)
    selected = st.multiselect("Select batters", options, default=[])
    histories_b = {
        label: batter_history[batter_label_to_id[label]]
        for label in selected
        if batter_label_to_id[label] in batter_history
    }
    st.plotly_chart(build_compressed_chart(histories_b, "PA", names),
                    use_container_width=True, key="batter_timeline")

with tab_p:
    options = sorted(pitcher_label_to_id)
    selected = st.multiselect("Select pitchers", options, default=[])
    histories_p = {
        label: pitcher_history[pitcher_label_to_id[label]]
        for label in selected
        if pitcher_label_to_id[label] in pitcher_history
    }
    st.plotly_chart(build_compressed_chart(histories_p, "BF", names),
                    use_container_width=True, key="pitcher_timeline")

# ── Park factors ──────────────────────────────────────────────────────────────
with st.expander("Park Factors (computed from data)"):
    st.caption(
        "wOBA park factors via the home/road method, regressed 50% toward 1.0. "
        ">1.00 = hitter-friendly (the ELO bar is raised there); <1.00 = pitcher-friendly. "
        "Each PA's expected wOBA is multiplied by its park factor before scoring."
    )
    pf_dict = compute_park_factors(df)
    pf_df = pd.DataFrame(
        [{"Season": s, "Park": t, "Park Factor": round(v, 3)} for (s, t), v in pf_dict.items()]
    ).sort_values(["Season", "Park Factor"], ascending=[True, False]).reset_index(drop=True)
    st.dataframe(pf_df, use_container_width=True, hide_index=True, height=400)

# ── Diagnostics ───────────────────────────────────────────────────────────────
with st.expander("Diagnostics"):
    # PA-weighted mean: the calibration check. Both pools are recentered to
    # 1500 (PA-weighted) at the end of run_elo, so these should read ~1500.0.
    def pa_weighted_mean(ratings: dict, pa_counts: pd.Series) -> float:
        total_w, total_wr = 0.0, 0.0
        for pid, r in ratings.items():
            w = pa_counts.get(pid, 0)
            total_w += w
            total_wr += r * w
        return total_wr / total_w if total_w else 0.0

    b_wmean = pa_weighted_mean(batter_ratings, batter_pa)
    p_wmean = pa_weighted_mean(pitcher_ratings, pitcher_pa)
    c1, c2, c3 = st.columns(3)
    c1.metric("PA-weighted batter ELO", f"{b_wmean:.1f}")
    c2.metric("PA-weighted pitcher ELO", f"{p_wmean:.1f}")
    c3.metric("League wOBA used", f"{league_woba:.4f}")
    st.caption("Both pools are recentered so the PA-weighted mean is 1500, "
               "keeping batters and pitchers on a comparable scale.")
