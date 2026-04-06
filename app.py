from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from icalendar import Calendar

st.set_page_config(page_title="Comparateur ICS - Vue agenda", layout="wide")


# ============================================================
# CONFIG
# ============================================================
DISPLAY_TZ = "Europe/Paris"
DAY_START_HOUR = 8
DAY_END_HOUR = 21
PIXELS_PER_HOUR = 90
HEADER_HEIGHT = 46
LEFT_TIME_COL_WIDTH = 70
MIN_EVENT_HEIGHT = 28


# ============================================================
# OUTILS
# ============================================================
def ensure_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, time.min)
    else:
        dt = pd.to_datetime(value).to_pydatetime()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def parse_ics_file(file_name: str, raw_bytes: bytes) -> pd.DataFrame:
    cal = Calendar.from_ical(raw_bytes)
    rows: List[Dict[str, Any]] = []

    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        dtstart = component.get("DTSTART")
        dtend = component.get("DTEND")
        summary = component.get("SUMMARY")
        location = component.get("LOCATION")
        description = component.get("DESCRIPTION")
        uid = component.get("UID")

        if dtstart is None:
            continue

        start = ensure_datetime(dtstart.dt)
        end = ensure_datetime(dtend.dt) if dtend is not None else start

        if end < start:
            end = start

        rows.append(
            {
                "calendar": file_name,
                "uid": safe_text(uid),
                "title": safe_text(summary) or "(sans titre)",
                "location": safe_text(location),
                "description": safe_text(description),
                "start_utc": start,
                "end_utc": end,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "calendar",
                "uid",
                "title",
                "location",
                "description",
                "start_utc",
                "end_utc",
            ]
        )

    df = pd.DataFrame(rows).sort_values(["start_utc", "end_utc", "title"]).reset_index(drop=True)
    return df


def enrich_local_columns(df: pd.DataFrame, tz_name: str = DISPLAY_TZ) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["start_local"] = pd.to_datetime(out["start_utc"], utc=True).dt.tz_convert(tz_name)
    out["end_local"] = pd.to_datetime(out["end_utc"], utc=True).dt.tz_convert(tz_name)
    out["start_day"] = out["start_local"].dt.date
    out["end_day"] = out["end_local"].dt.date
    return out


def split_multi_day_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coupe les événements multi-jours pour une vue journalière.
    """
    if df.empty:
        return df.copy()

    pieces: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        start = row["start_local"]
        end = row["end_local"]

        current_day = start.date()
        last_day = end.date()

        while current_day <= last_day:
            day_start = pd.Timestamp(datetime.combine(current_day, time.min), tz=start.tz)
            day_end = day_start + pd.Timedelta(days=1)

            seg_start = max(start, day_start)
            seg_end = min(end, day_end)

            if seg_start < seg_end:
                piece = row.to_dict()
                piece["day"] = current_day
                piece["segment_start"] = seg_start
                piece["segment_end"] = seg_end
                piece["segment_start_str"] = seg_start.strftime("%H:%M")
                piece["segment_end_str"] = seg_end.strftime("%H:%M")
                piece["duration_min"] = int((seg_end - seg_start).total_seconds() / 60)
                pieces.append(piece)

            current_day += timedelta(days=1)

    return pd.DataFrame(pieces)


def intervals_overlap(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def assign_columns_for_overlaps(events: pd.DataFrame) -> pd.DataFrame:
    """
    Pour une colonne calendrier donnée et un jour donné :
    calcule des sous-colonnes internes si des événements se chevauchent.
    """
    if events.empty:
        out = events.copy()
        out["overlap_col"] = 0
        out["overlap_count"] = 1
        return out

    events = events.sort_values(["segment_start", "segment_end", "title"]).reset_index(drop=True).copy()

    active = []
    overlap_col = []
    overlap_group_counts = [1] * len(events)

    for i, row in events.iterrows():
        active = [(idx, col, end) for idx, col, end in active if end > row["segment_start"]]
        used_cols = {col for _, col, _ in active}

        col = 0
        while col in used_cols:
            col += 1

        active.append((i, col, row["segment_end"]))
        overlap_col.append(col)

        current_count = max(c for _, c, _ in active) + 1
        involved_indices = [idx for idx, _, _ in active]
        for idx in involved_indices:
            overlap_group_counts[idx] = max(overlap_group_counts[idx], current_count)

    events["overlap_col"] = overlap_col
    events["overlap_count"] = overlap_group_counts
    return events


def detect_cross_calendar_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les conflits entre calendriers différents sur le même jour.
    """
    if df.empty:
        out = df.copy()
        out["has_conflict"] = False
        out["conflict_with"] = ""
        return out

    out = df.copy().reset_index(drop=True)
    out["has_conflict"] = False
    out["conflict_with"] = ""

    for day in sorted(out["day"].unique()):
        day_df = out[out["day"] == day]

        for i in day_df.index:
            for j in day_df.index:
                if j <= i:
                    continue

                a = out.loc[i]
                b = out.loc[j]

                if a["calendar"] == b["calendar"]:
                    continue

                if intervals_overlap(a["segment_start"], a["segment_end"], b["segment_start"], b["segment_end"]):
                    out.loc[i, "has_conflict"] = True
                    out.loc[j, "has_conflict"] = True

                    a_conf = set(filter(None, str(out.loc[i, "conflict_with"]).split(" | ")))
                    b_conf = set(filter(None, str(out.loc[j, "conflict_with"]).split(" | ")))
                    a_conf.add(str(b["calendar"]))
                    b_conf.add(str(a["calendar"]))
                    out.loc[i, "conflict_with"] = " | ".join(sorted(a_conf))
                    out.loc[j, "conflict_with"] = " | ".join(sorted(b_conf))

    return out


def format_day_fr(d: date) -> str:
    jours = [
        "Lundi", "Mardi", "Mercredi", "Jeudi",
        "Vendredi", "Samedi", "Dimanche"
    ]
    mois = [
        "", "janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre"
    ]
    return f"{jours[d.weekday()]} {d.day} {mois[d.month]} {d.year}"


def day_bounds(day_value: date, tz_name: str = DISPLAY_TZ) -> tuple[pd.Timestamp, pd.Timestamp]:
    tz = pd.Timestamp.now(tz=tz_name).tz
    start = pd.Timestamp(datetime.combine(day_value, time.min), tz=tz)
    end = start + pd.Timedelta(days=1)
    return start, end


def crop_event_to_visible_hours(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    visible_start = start_dt.normalize() + pd.Timedelta(hours=DAY_START_HOUR)
    visible_end = start_dt.normalize() + pd.Timedelta(hours=DAY_END_HOUR)

    cropped_start = max(start_dt, visible_start)
    cropped_end = min(end_dt, visible_end)

    if cropped_end <= cropped_start:
        return None

    return cropped_start, cropped_end


def event_to_style(start_dt: pd.Timestamp, end_dt: pd.Timestamp, overlap_col: int, overlap_count: int) -> dict:
    visible = crop_event_to_visible_hours(start_dt, end_dt)
    if visible is None:
        return {}

    cropped_start, cropped_end = visible

    total_minutes_from_top = (cropped_start.hour * 60 + cropped_start.minute) - DAY_START_HOUR * 60
    duration_minutes = (cropped_end - cropped_start).total_seconds() / 60

    top = total_minutes_from_top * PIXELS_PER_HOUR / 60
    height = max(MIN_EVENT_HEIGHT, duration_minutes * PIXELS_PER_HOUR / 60)

    inner_gap = 6
    width_pct = 100 / max(1, overlap_count)
    left_pct = overlap_col * width_pct

    return {
        "top": top,
        "height": height,
        "left_pct": left_pct,
        "width_pct": width_pct,
        "inner_gap": inner_gap,
    }


def render_agenda_for_day(day_df: pd.DataFrame, calendars: List[str]) -> str:
    total_height = (DAY_END_HOUR - DAY_START_HOUR) * PIXELS_PER_HOUR
    hour_lines = list(range(DAY_START_HOUR, DAY_END_HOUR + 1))

    html_parts = []

    html_parts.append(
        f"""
        <style>
            .agenda-shell {{
                width: 100%;
                border: 1px solid #d9d9d9;
                border-radius: 16px;
                overflow: hidden;
                background: #ffffff;
                box-shadow: 0 2px 10px rgba(0,0,0,0.06);
                margin-bottom: 22px;
            }}

            .agenda-header {{
                display: grid;
                grid-template-columns: {LEFT_TIME_COL_WIDTH}px repeat({len(calendars)}, 1fr);
                height: {HEADER_HEIGHT}px;
                border-bottom: 1px solid #ececec;
                background: #fafafa;
                font-weight: 600;
                color: #222;
            }}

            .agenda-header-cell {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 8px;
                border-left: 1px solid #ececec;
                font-size: 14px;
                text-align: center;
            }}

            .agenda-header-time {{
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 13px;
                color: #777;
            }}

            .agenda-body {{
                display: grid;
                grid-template-columns: {LEFT_TIME_COL_WIDTH}px repeat({len(calendars)}, 1fr);
                position: relative;
                height: {total_height}px;
            }}

            .time-col {{
                position: relative;
                background: #fff;
                border-right: 1px solid #ececec;
            }}

            .time-label {{
                position: absolute;
                right: 10px;
                transform: translateY(-50%);
                font-size: 12px;
                color: #666;
                font-weight: 500;
            }}

            .calendar-col {{
                position: relative;
                border-left: 1px solid #ececec;
                background:
                    repeating-linear-gradient(
                        to bottom,
                        #ffffff 0px,
                        #ffffff {PIXELS_PER_HOUR - 1}px,
                        #efefef {PIXELS_PER_HOUR}px
                    );
            }}

            .event-card {{
                position: absolute;
                border-left: 4px solid #d97706;
                background: #f8e8df;
                border-radius: 8px;
                padding: 6px 8px;
                overflow: hidden;
                font-size: 12px;
                line-height: 1.2;
                box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                color: #1f2937;
            }}

            .event-card.conflict {{
                border-left-color: #dc2626;
                background: #fde8e8;
            }}

            .event-badge {{
                display: inline-block;
                padding: 2px 7px;
                border-radius: 999px;
                background: #e58d38;
                color: #111;
                font-size: 10px;
                font-weight: 700;
                margin-bottom: 6px;
            }}

            .event-title {{
                font-weight: 700;
                font-size: 13px;
                margin-bottom: 4px;
            }}

            .event-time {{
                font-size: 11px;
                color: #374151;
                margin-bottom: 3px;
            }}

            .event-location {{
                font-size: 11px;
                color: #374151;
            }}

            .event-conflict {{
                margin-top: 5px;
                font-size: 10px;
                color: #b91c1c;
                font-weight: 700;
            }}
        </style>
        """
    )

    html_parts.append('<div class="agenda-shell">')

    html_parts.append('<div class="agenda-header">')
    html_parts.append('<div class="agenda-header-time">Heure</div>')
    for cal in calendars:
        html_parts.append(f'<div class="agenda-header-cell">{cal}</div>')
    html_parts.append("</div>")

    html_parts.append('<div class="agenda-body">')

    # colonne heures
    html_parts.append('<div class="time-col">')
    for hour in hour_lines:
        top = (hour - DAY_START_HOUR) * PIXELS_PER_HOUR
        if 0 <= top <= total_height:
            html_parts.append(
                f'<div class="time-label" style="top:{top}px;">{hour:02d}:00</div>'
            )
    html_parts.append("</div>")

    # colonnes agendas
    for cal in calendars:
        cal_df = day_df[day_df["calendar"] == cal].copy()
        cal_df = assign_columns_for_overlaps(cal_df)

        html_parts.append('<div class="calendar-col">')

        for _, event in cal_df.iterrows():
            style = event_to_style(
                event["segment_start"],
                event["segment_end"],
                int(event["overlap_col"]),
                int(event["overlap_count"]),
            )

            if not style:
                continue

            css_class = "event-card conflict" if bool(event["has_conflict"]) else "event-card"

            conflict_html = ""
            if bool(event["has_conflict"]) and str(event["conflict_with"]).strip():
                conflict_html = f'<div class="event-conflict">Conflit: {event["conflict_with"]}</div>'

            location_html = ""
            if str(event["location"]).strip():
                location_html = f'<div class="event-location">📍 {event["location"]}</div>'

            html_parts.append(
                f"""
                <div class="{css_class}"
                     style="
                        top:{style['top']}px;
                        height:{style['height']}px;
                        left:calc({style['left_pct']}% + {style['inner_gap'] / 2}px);
                        width:calc({style['width_pct']}% - {style['inner_gap']}px);
                     ">
                    <div class="event-badge">Cours</div>
                    <div class="event-title">{event["title"]}</div>
                    <div class="event-time">{event["segment_start_str"]} → {event["segment_end_str"]}</div>
                    {location_html}
                    {conflict_html}
                </div>
                """
            )

        html_parts.append("</div>")

    html_parts.append("</div>")
    html_parts.append("</div>")

    return "".join(html_parts)


# ============================================================
# CHARGEMENT
# ============================================================
st.title("Comparateur de calendriers ICS")
st.caption("Vue agenda journalière, avec plusieurs calendriers côte à côte.")

uploaded_files = st.file_uploader(
    "Ajoute 2 fichiers .ics ou plus",
    type=["ics"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Charge plusieurs fichiers .ics pour afficher la vue agenda.")
    st.stop()

dfs = []
errors = []

for file in uploaded_files:
    try:
        content = file.read()
        df_file = parse_ics_file(file.name, content)
        dfs.append(df_file)
    except Exception as exc:
        errors.append((file.name, str(exc)))

if errors:
    st.warning("Certains fichiers n'ont pas pu être lus :")
    for file_name, err in errors:
        st.write(f"- {file_name}: {err}")

if not dfs:
    st.error("Aucun calendrier exploitable.")
    st.stop()

events_df = pd.concat(dfs, ignore_index=True)

if events_df.empty:
    st.warning("Aucun événement trouvé dans les fichiers fournis.")
    st.stop()

events_df = enrich_local_columns(events_df, DISPLAY_TZ)
events_df = split_multi_day_events(events_df)
events_df = detect_cross_calendar_conflicts(events_df)

all_calendars = sorted(events_df["calendar"].unique())
all_days = sorted(events_df["day"].unique())

if not all_days:
    st.warning("Aucun jour disponible.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Filtres")

selected_calendars = st.sidebar.multiselect(
    "Calendriers à afficher",
    options=all_calendars,
    default=all_calendars,
)

min_day = min(all_days)
max_day = max(all_days)

date_range = st.sidebar.date_input(
    "Période",
    value=(min_day, max_day),
    min_value=min_day,
    max_value=max_day,
)

show_only_conflicts = st.sidebar.checkbox("Afficher seulement les jours avec conflit", value=False)

start_hour = st.sidebar.slider("Heure de début visible", min_value=0, max_value=23, value=DAY_START_HOUR)
end_hour = st.sidebar.slider("Heure de fin visible", min_value=1, max_value=24, value=DAY_END_HOUR)

if start_hour >= end_hour:
    st.sidebar.error("L'heure de début doit être inférieure à l'heure de fin.")
    st.stop()

DAY_START_HOUR = start_hour
DAY_END_HOUR = end_hour

if isinstance(date_range, tuple) and len(date_range) == 2:
    selected_start_day, selected_end_day = date_range
else:
    selected_start_day, selected_end_day = min_day, max_day

filtered = events_df[
    (events_df["calendar"].isin(selected_calendars))
    & (events_df["day"] >= selected_start_day)
    & (events_df["day"] <= selected_end_day)
].copy()

if filtered.empty:
    st.warning("Aucun événement avec les filtres choisis.")
    st.stop()

days_to_show = sorted(filtered["day"].unique())

if show_only_conflicts:
    days_to_show = [
        d for d in days_to_show
        if filtered.loc[filtered["day"] == d, "has_conflict"].any()
    ]

if not days_to_show:
    st.success("Aucun jour avec conflit dans la période sélectionnée.")
    st.stop()


# ============================================================
# METRICS
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fichiers chargés", len(uploaded_files))
col2.metric("Calendriers affichés", len(selected_calendars))
col3.metric("Jours affichés", len(days_to_show))
col4.metric("Événements visibles", len(filtered))


# ============================================================
# NAV JOUR
# ============================================================
selected_day = st.selectbox(
    "Jour à afficher",
    options=days_to_show,
    format_func=lambda d: format_day_fr(d),
)

day_df = filtered[filtered["day"] == selected_day].copy()

st.markdown(f"## {format_day_fr(selected_day)}")
st.caption("Chaque colonne correspond à un agenda différent.")

agenda_html = render_agenda_for_day(day_df, selected_calendars)
st.components.v1.html(agenda_html, height=(DAY_END_HOUR - DAY_START_HOUR) * PIXELS_PER_HOUR + HEADER_HEIGHT + 30, scrolling=True)


# ============================================================
# DÉTAILS TEXTE
# ============================================================
with st.expander("Afficher aussi la liste détaillée des événements du jour"):
    display_df = day_df[
        [
            "calendar",
            "title",
            "segment_start_str",
            "segment_end_str",
            "location",
            "has_conflict",
            "conflict_with",
        ]
    ].sort_values(["calendar", "segment_start_str", "segment_end_str", "title"])
    st.dataframe(display_df, use_container_width=True)