# Baseball ELO Ratings

An interactive ELO rating system for MLB batters and pitchers, built on pitch-by-pitch
Statcast data. Every plate appearance is a zero-sum contest: the batter and pitcher
exchange rating points based on the outcome, weighted by how much it mattered.

## Features

- **wOBA-weighted outcomes** — a home run moves ratings far more than a walk, using each
  PA's wOBA value rather than a simple on-base/out binary.
- **Park factors** — computed per season from the data itself (home/road method), so a hit
  in Coors counts less than one in a pitcher's park.
- **Dynamic K-factor** — rating updates scale with leverage (win-probability impact) and a
  player's career length (new players move faster), with no manual tuning.
- **Self-calibrating** — each pool is recentered so batters and pitchers both average 1500
  on a plate-appearance-weighted basis, regardless of the league's run environment.
- **Single-season or career** — ratings reset each year, or carry across seasons.
- **Leaderboards** — sortable by End / Avg / Peak / Worst ELO and Range, filterable by team.
- **Cross-year matchup predictor** — pit a 2016 hitter against a 2026 pitcher in any
  season's run environment and see the expected wOBA.
- **Per-PA rating timelines** — chart any player's rating across a season, with off-days
  compressed and each plate appearance hoverable.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first load of an uncached season downloads Statcast data (a few minutes) and caches it
locally under `cache/`. Subsequent loads are instant.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — leaderboards, matchup predictor, charts |
| `elo.py` | ELO engine, park factors, leaderboard builder |
| `data_fetch.py` | Statcast download, caching, player-name resolution |

## Data

Statcast data via [pybaseball](https://github.com/jldbc/pybaseball). Coverage begins 2015.
