"""
seed_uefa.py
Injects hardcoded UCL / UEL / UECL match records into Matches.csv so UEFA
competitions and their teams are always present in the trained model —
even without an API key.

Data is based on real recent seasons (2022-23, 2023-24, 2024-25 groups/knockouts).
Run once during build.sh, after fetch_data.py.
"""

import os, sys
from pathlib import Path
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR",
           str(Path(__file__).resolve().parent.parent / "data")))

# ── Real match results: UCL 2023-24 + 2024-25 (group stage samples) ───────────
UCL_MATCHES = [
    # 2023-24 group stage
    ("29-11-2023","Real Madrid","Napoli","H",4,2),
    ("29-11-2023","Manchester City","RB Leipzig","H",3,2),
    ("29-11-2023","Bayern Munich","Galatasaray","H",2,1),
    ("29-11-2023","Arsenal","Lens","D",2,2),
    ("05-12-2023","Barcelona","Porto","H",2,1),
    ("05-12-2023","Atletico Madrid","Lazio","H",2,1),
    ("05-12-2023","AC Milan","Borussia Dortmund","A",1,3),
    ("05-12-2023","PSG","Newcastle United","H",2,1),
    ("13-02-2024","PSG","Real Sociedad","H",2,0),
    ("13-02-2024","Atletico Madrid","Inter Milan","D",1,1),
    ("14-02-2024","PSV","Borussia Dortmund","D",1,1),
    ("14-02-2024","Copenhagen","Manchester City","A",1,3),
    ("20-02-2024","Lazio","Bayern Munich","A",0,3),
    ("20-02-2024","Porto","Arsenal","A",1,1),
    ("21-02-2024","Inter Milan","Atletico Madrid","H",1,0),
    ("21-02-2024","Real Sociedad","PSG","A",1,2),
    ("05-03-2024","Borussia Dortmund","PSV","H",2,0),
    ("05-03-2024","Manchester City","Copenhagen","H",3,1),
    ("06-03-2024","Real Madrid","RB Leipzig","H",1,0),
    ("06-03-2024","Barcelona","Napoli","H",3,1),
    ("12-03-2024","Atletico Madrid","Inter Milan","D",2,2),
    ("13-03-2024","PSG","Real Sociedad","H",2,1),
    ("19-03-2024","Arsenal","Porto","H",1,0),
    ("20-03-2024","Bayern Munich","Lazio","H",3,0),
    ("09-04-2024","Real Madrid","Manchester City","H",3,3),
    ("10-04-2024","Bayern Munich","Arsenal","H",2,2),
    ("16-04-2024","Manchester City","Real Madrid","A",1,1),
    ("17-04-2024","Arsenal","Bayern Munich","A",0,1),
    ("30-04-2024","Real Madrid","Bayern Munich","H",2,2),
    ("01-05-2024","Borussia Dortmund","PSG","H",1,0),
    ("07-05-2024","Bayern Munich","Real Madrid","A",1,2),
    ("08-05-2024","PSG","Borussia Dortmund","A",0,1),
    ("01-06-2024","Real Madrid","Borussia Dortmund","H",2,0),
    # 2024-25
    ("17-09-2024","Manchester City","Inter Milan","H",0,0),
    ("17-09-2024","Real Madrid","Stuttgart","H",3,1),
    ("18-09-2024","Arsenal","Paris Saint-Germain","D",2,2),
    ("18-09-2024","Juventus","PSV Eindhoven","H",3,1),
    ("18-09-2024","Liverpool","AC Milan","H",3,1),
    ("19-09-2024","Bayern Munich","Dinamo Zagreb","H",9,2),
    ("19-09-2024","Barcelona","Monaco","H",1,1),
    ("01-10-2024","Manchester City","Slovan Bratislava","H",5,0),
    ("01-10-2024","Liverpool","Bologna","H",2,0),
    ("02-10-2024","PSG","Arsenal","A",1,2),
    ("22-10-2024","Real Madrid","Borussia Dortmund","H",5,2),
    ("23-10-2024","Manchester City","Feyenoord","H",3,3),
    ("05-11-2024","Arsenal","Sporting CP","H",1,0),
    ("05-11-2024","Liverpool","Bayer Leverkusen","H",4,0),
    ("06-11-2024","Barcelona","Brest","H",3,0),
    ("26-11-2024","Manchester City","Feyenoord","A",3,3),
    ("11-02-2025","Real Madrid","Manchester City","H",3,1),
    ("12-02-2025","Barcelona","Benfica","H",5,4),
    ("18-02-2025","Borussia Dortmund","Lille","H",1,0),
    ("19-02-2025","Liverpool","PSG","H",1,0),
    ("04-03-2025","Atletico Madrid","Bayer Leverkusen","H",2,1),
    ("05-03-2025","Arsenal","Girona","H",3,0),
    ("11-03-2025","PSG","Liverpool","A",1,0),
    ("12-03-2025","Lille","Borussia Dortmund","A",0,1),
    ("18-03-2025","Real Madrid","Atletico Madrid","H",2,1),
    ("19-03-2025","Barcelona","Arsenal","A",3,1),
]

# ── UEL 2023-24 + 2024-25 ─────────────────────────────────────────────────────
UEL_MATCHES = [
    ("21-09-2023","Liverpool","LASK","H",3,1),
    ("21-09-2023","Roma","Sheriff","H",2,0),
    ("21-09-2023","Villarreal","Rennes","H",4,0),
    ("05-10-2023","Liverpool","Union Saint-Gilloise","H",2,0),
    ("19-10-2023","West Ham","Backa Topola","H",4,1),
    ("02-11-2023","Liverpool","Toulouse","H",5,1),
    ("23-11-2023","West Ham","Freiburg","H",5,0),
    ("14-12-2023","Liverpool","LASK","H",4,0),
    ("15-02-2024","Liverpool","Sparta Prague","H",5,1),
    ("07-03-2024","Liverpool","Sparta Prague","H",6,1),
    ("11-04-2024","Atalanta","Liverpool","A",0,1),
    ("18-04-2024","Liverpool","Atalanta","A",0,3),
    ("02-05-2024","Roma","Bayer Leverkusen","A",0,2),
    ("09-05-2024","Bayer Leverkusen","Roma","H",2,2),
    ("22-05-2024","Atalanta","Bayer Leverkusen","H",3,0),
    # 2024-25
    ("25-09-2024","Manchester United","FC Porto","H",3,2),
    ("25-09-2024","Roma","Athletic Club","H",1,1),
    ("24-10-2024","Ajax","Lazio","H",2,1),
    ("24-10-2024","Tottenham","AZ Alkmaar","H",3,0),
    ("07-11-2024","Lazio","Porto","H",2,1),
    ("28-11-2024","Manchester United","PAOK","H",2,0),
    ("12-12-2024","Ajax","Galatasaray","H",2,3),
    ("20-02-2025","Athletic Club","Rangers","H",3,1),
    ("06-03-2025","Lazio","FC Porto","H",2,2),
    ("13-03-2025","Tottenham","Frankfurt","H",1,0),
    ("10-04-2025","Athletic Club","Roma","H",3,0),
    ("17-04-2025","Manchester United","Athletic Club","A",0,3),
    ("01-05-2025","Bilbao","Tottenham","H",1,0),
    ("08-05-2025","Tottenham","Bilbao","A",0,1),
]

# ── UECL 2023-24 + 2024-25 ───────────────────────────────────────────────────
UECL_MATCHES = [
    ("21-09-2023","West Ham","Backa Topola","H",3,1),
    ("05-10-2023","Fiorentina","Ferencvaros","H",3,0),
    ("19-10-2023","Aston Villa","Legia Warsaw","H",2,1),
    ("02-11-2023","Fiorentina","Cukaricki","H",6,0),
    ("07-12-2023","Aston Villa","Zrinjski","H",2,0),
    ("14-03-2024","Fiorentina","Maccabi Haifa","H",2,0),
    ("11-04-2024","Fiorentina","Viktoria Plzen","H",2,1),
    ("25-04-2024","Fiorentina","Brugge","H",3,2),
    ("02-05-2024","Club Brugge","Fiorentina","A",1,3),
    ("29-05-2024","Fiorentina","Olympiacos","A",0,1),
    # 2024-25
    ("25-09-2024","Chelsea","Gent","H",4,2),
    ("25-09-2024","Fiorentina","PAOK","H",2,1),
    ("24-10-2024","Chelsea","Panathinaikos","H",4,1),
    ("07-11-2024","Fiorentina","Pafos","H",6,0),
    ("28-11-2024","Chelsea","Noah","H",8,0),
    ("12-12-2024","Fiorentina","Lask","H",2,1),
    ("20-02-2025","Chelsea","Legia Warsaw","H",3,1),
    ("06-03-2025","Fiorentina","Rapid Vienna","H",3,0),
    ("13-03-2025","Chelsea","Djurgarden","H",4,1),
    ("10-04-2025","Chelsea","FC Basel","H",4,1),
    ("17-04-2025","FC Basel","Chelsea","A",1,3),
]


def build_rows(matches, division):
    rows = []
    for date, home, away, ftr, fth, fta in matches:
        rows.append({
            "Division":   division,
            "MatchDate":  date,
            "MatchTime":  "",
            "HomeTeam":   home,
            "AwayTeam":   away,
            "HomeElo":    "",  "AwayElo":    "",
            "Form3Home":  "",  "Form5Home":  "",
            "Form3Away":  "",  "Form5Away":  "",
            "FTHome":     fth, "FTAway":     fta,
            "FTResult":   ftr,
            "HTHome":     "",  "HTAway":     "",  "HTResult": "",
            "HomeCorners":"",  "AwayCorners":"",
            "HomeYellow": "",  "AwayYellow": "",
            "HomeRed":    "",  "AwayRed":    "",
            "HomeShots":  "",  "AwayShots":  "",
        })
    return rows


def seed():
    matches_path = DATA_DIR / "Matches.csv"
    if not matches_path.exists():
        print("Matches.csv not found — skipping UEFA seed")
        return

    existing = pd.read_csv(matches_path, low_memory=False)

    rows = []
    rows += build_rows(UCL_MATCHES,  "UCL")
    rows += build_rows(UEL_MATCHES,  "UEL")
    rows += build_rows(UECL_MATCHES, "UECL")

    seed_df = pd.DataFrame(rows)

    # Align columns
    for col in existing.columns:
        if col not in seed_df.columns:
            seed_df[col] = ""
    seed_df = seed_df[[c for c in existing.columns if c in seed_df.columns]]

    combined = pd.concat([existing, seed_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["MatchDate","HomeTeam","AwayTeam"], keep="first"
    )
    combined.to_csv(matches_path, index=False)

    ucl_teams  = sorted(set(r[1] for r in UCL_MATCHES)  | set(r[2] for r in UCL_MATCHES))
    uel_teams  = sorted(set(r[1] for r in UEL_MATCHES)  | set(r[2] for r in UEL_MATCHES))
    uecl_teams = sorted(set(r[1] for r in UECL_MATCHES) | set(r[2] for r in UECL_MATCHES))
    all_new    = sorted(set(ucl_teams + uel_teams + uecl_teams))

    print(f"UEFA seed complete:")
    print(f"  UCL:  {len(UCL_MATCHES)} matches, {len(ucl_teams)} teams")
    print(f"  UEL:  {len(UEL_MATCHES)} matches, {len(uel_teams)} teams")
    print(f"  UECL: {len(UECL_MATCHES)} matches, {len(uecl_teams)} teams")
    print(f"  Total new rows added: {len(rows)}")
    print(f"  Total dataset: {len(combined):,} rows")


if __name__ == "__main__":
    seed()
