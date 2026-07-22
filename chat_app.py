"""
Minimal Streamlit chat UI for the Lake County contract due-diligence pipeline.

Full corpus: 123 Tier 1/1.5 documents with full field extraction, 226 Tier 2/4
existence-only records, 349 total rows in contracts_master. The ChromaDB
clause store covers all 123 Tier 1/1.5 docs.

Usage: py -m streamlit run chat_app.py
"""

import anthropic
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import voyageai

from query_router import answer_question, get_chroma_collection

# Chart cleaning: label values that mean "no real data here" and shouldn't
# become a bar, plus friendlier display names for raw enum/status values.
NULL_LABEL_VALUES = {None, "null", ""}
LABEL_MAP = {
    "expired_but_renewal_on_file": "Expired (renewal on file)",
    "expiration_not_stated": "Expiration not stated",
    "not_yet_effective": "Not yet effective",
    "not_stated": "Not stated",
    "option_to_renew": "Option to renew",
    "auto_renew_unless_cancelled": "Auto-renew",
    "competitive_bid": "Competitive bid",
    "sole_source": "Sole source",
    "cooperative_purchasing_vehicle": "Cooperative purchasing",
    "base_agreement": "Base agreement",
    "rate_based": "Rate-based",
    "fixed_fee": "Fixed fee",
    "per_unit": "Per unit",
    "time_and_materials": "T&M",
    "not_applicable": "N/A",
}
# Validated categorical order (blue/green/magenta/yellow/aqua) -- passes CVD-
# safety checks in this exact order (see dataviz skill's palette.md); used for
# charts that aren't risk/commitment in nature (fee-structure mix, doc-role
# breakdown). A near-black previously sat in this cycle as if it were a hue,
# which read as an off/broken bar next to real colors -- removed.
CHART_COLORS = ["2A78D6", "008300", "E87BA4", "EDA100", "1BAF7A"]
# Single-color fallback for long leaderboard-style charts (>5 categories, e.g.
# "top vendors") -- kept as the deck's teal accent rather than reusing the
# first categorical slot, so a >5-category chart doesn't look like a
# 1-category categorical chart cut off after one color.
LEADERBOARD_COLOR = "0D9488"

# Deck color system (slide3_v5): risk/exposure categories are rose, committed-
# spend/confirmed categories are teal -- deliberately just these two colors,
# no amber/blue/green, so the dashboard stays visually consistent with the
# deck if someone flips between both. Only applies when EVERY label in a
# chart is one of these two buckets (see _is_risk_commitment_chart) -- charts
# that aren't inherently risk-vs-commitment (fee structure mix, vendor counts,
# doc_role breakdown) keep the CHART_COLORS cycle above, since forcing an
# arbitrary rose/teal binary onto categories like "Rate-based"/"Fixed fee"
# would misrepresent them as risk/commitment when they aren't.
RISK_COLOR = "#E11D48"
COMMITMENT_COLOR = "#0D9488"
RISK_LABELS = {
    "uncontracted", "expired", "expired (renewal on file)", "immediate action",
    "action needed", "unknown", "expiration not stated", "not stated",
    "renewal status uncertain",
}
COMMITMENT_LABELS = {"active", "confirmed", "renewed", "committed"}

# Fixed colors for the base-contract-status chart specifically -- flat red for
# "Uncontracted" read as alarming for what's often just an unrenewed contract,
# not an active crisis, so it gets a calmer amber; "Renewal status uncertain"
# isn't a risk/commitment call at all (it means "we don't know yet"), so it
# gets neutral grey rather than being forced into the rose/teal binary. Active
# keeps the same teal as the rest of the deck's commitment color. Applied
# whenever every label in a chart is one of these three -- covers both the
# 3-bar base-status chart (Active/Uncontracted/Renewal status uncertain, after
# "Expiration not stated" is split out below) and the plain 2-bar Active-vs-
# Uncontracted aggregate.
STATUS_COLORS = {
    "active": COMMITMENT_COLOR,
    "uncontracted": "#D97706",
    "renewal status uncertain": "#9CA3AF",
}

# Grounded in the actual corpus (see CLAUDE.md "Known Fixes" #9): most
# "Expiration not stated" contracts are a genuine fact about the agreement,
# not a pipeline gap -- e.g. evergreen/ongoing service contracts, or ones
# whose termination is governed by a notice period rather than a fixed end
# date. "Renewal status uncertain" contracts have a renewal letter on file
# that isn't confident enough to resolve to a specific still-active date.
EXPIRATION_NOT_STATED_NOTE = (
    "**{n} base contracts don't state a fixed expiration date.** In most cases this is a real fact "
    "about the agreement, not a data gap -- e.g. an evergreen/ongoing service contract, or one where "
    "termination is governed by a notice period (\"either party may terminate with 30 days' written "
    "notice\") rather than a stated end date."
)
RENEWAL_UNCERTAIN_NOTE = (
    "**{n} base contracts have a renewal letter on file that isn't confident enough to confirm a "
    "specific still-active date** -- the renewal reference is stale or its coverage period couldn't "
    "be resolved. These need manual verification before being counted as Active or Uncontracted."
)


def _is_risk_commitment_chart(labels: list[str]) -> bool:
    return bool(labels) and all(lbl.lower() in RISK_LABELS or lbl.lower() in COMMITMENT_LABELS for lbl in labels)


def _risk_commitment_colors(labels: list[str]) -> list[str]:
    return [RISK_COLOR if lbl.lower() in RISK_LABELS else COMMITMENT_COLOR for lbl in labels]


def split_base_status_rows(rows: list[tuple]) -> tuple[list[tuple], list[tuple]]:
    """Detects the 4-bucket base-contract-status shape (Active / Uncontracted /
    Expiration not stated / Renewal status uncertain) and pulls the latter two
    out as footnotes rather than bars -- they aren't risk states in the same
    sense as Active/Uncontracted, and charting all 4 buried the Active-vs-
    Uncontracted signal the user actually cares about. Returns
    (chart_rows, footnote_rows); footnote_rows is empty for any other shape."""
    labels_lower = {str(label).strip().lower() for label, _ in rows}
    if not {"expiration not stated", "renewal status uncertain"} <= labels_lower:
        return rows, []
    chart_rows, footnotes = [], []
    for label, value in rows:
        if str(label).strip().lower() in ("expiration not stated", "renewal status uncertain"):
            footnotes.append((label, value))
        else:
            chart_rows.append((label, value))
    return chart_rows, footnotes


def clean_chart_rows(rows: list[tuple]) -> list[tuple[str, float]]:
    """Drop null/placeholder-label rows, apply LABEL_MAP, sort by value descending."""
    cleaned = []
    for label, value in rows:
        label_str = "" if label is None else str(label).strip()
        if label in NULL_LABEL_VALUES or label_str == "" or label_str == "not_applicable_doc_type":
            continue
        if label_str.startswith("not_stated - see referenced"):
            continue
        cleaned.append((LABEL_MAP.get(label_str, label_str), value))
    # float(), not the raw cell -- is_chart_candidate only checks float()-ability, not
    # actual type, so a value column with TEXT storage affinity (e.g. "9" vs "31") would
    # otherwise sort lexicographically and put the wrong bar on top.
    cleaned.sort(key=lambda t: float(t[1]), reverse=True)
    return cleaned


def render_bar_chart(rows: list[tuple]) -> None:
    cleaned = clean_chart_rows(rows)
    if not cleaned:
        st.info("No chartable data after filtering out null/placeholder values.")
        return
    labels, values = zip(*cleaned)
    n = len(labels)
    if all(lbl.lower() in STATUS_COLORS for lbl in labels):
        colors = [STATUS_COLORS[lbl.lower()] for lbl in labels]
    elif _is_risk_commitment_chart(labels):
        colors = _risk_commitment_colors(labels)
    else:
        colors = [f"#{CHART_COLORS[i % len(CHART_COLORS)]}" for i in range(n)] if n <= 5 else [f"#{LEADERBOARD_COLOR}"] * n
    text = [str(int(v)) if float(v).is_integer() else str(v) for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=text, textposition="outside",
        textfont=dict(size=13, color="#374151"),
    ))
    # Per-bar height shrinks as the bar count grows, so a long leaderboard-style
    # chart (e.g. "top vendors", 15 rows) still fits in a single window instead
    # of scrolling off-screen -- a small chart (2-5 categories, e.g. active/
    # uncontracted or fee-structure mix) keeps the original generous spacing,
    # since it already reads clearly and doesn't need to shrink.
    if n <= 5:
        bar_height = 80
    elif n <= 10:
        bar_height = 40
    else:
        bar_height = 28
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=60, t=10, b=10),
        height=max(200, n * bar_height),
        xaxis=dict(visible=False, showgrid=False),
        # autorange="reversed" so the largest (first, since sorted descending)
        # bar renders at the top -- Plotly's default y-axis order for a
        # horizontal bar chart is bottom-to-top in list order, which would
        # otherwise put the smallest value on top and defeat the sort.
        yaxis=dict(tickfont=dict(size=13, color="#111827"), showgrid=False, autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


SUGGESTED_QUERIES = [
    "What's the status of our base contracts?",
    "Show active contract breakdown",
    "Which contracts are expiring soon and need action?",
    "Who are our top vendors by number of contracts?",
    "How are our base contracts structured by fee type?",
]

# Separate group: these route to the vector/clause-search path rather than
# SQL -- retrieving actual contract language, not structured field values.
# Kept apart from SUGGESTED_QUERIES above so the sidebar signals the two
# distinct capabilities (structured lookups vs. clause-level retrieval)
# instead of listing all 7 as if they were interchangeable.
SUGGESTED_CLAUSE_QUERIES = [
    "What does the indemnification clause say?",
    "What is the insurance clause for the Motorola base contract?",
]

st.set_page_config(page_title="Lake County Contracts Chat", page_icon="\U0001F4C4")
st.title("Lake County Contracts Chat")
st.caption(
    "Ask about vendors, dates, dollar amounts, renewal risk, and contract clause "
    "language across Lake County's procurement contracts."
)

with st.sidebar:
    dev_mode = st.checkbox("Dev mode (show step timings)", value=False)
    st.subheader("Try asking:")
    for i, suggested_query in enumerate(SUGGESTED_QUERIES):
        if st.button(suggested_query, key=f"suggested_query_{i}", use_container_width=True):
            st.session_state.pending_question = suggested_query
            st.rerun()
    st.subheader("You can also retrieve contract clauses:")
    for i, clause_query in enumerate(SUGGESTED_CLAUSE_QUERIES):
        if st.button(clause_query, key=f"suggested_clause_query_{i}", use_container_width=True):
            st.session_state.pending_question = clause_query
            st.rerun()


@st.cache_resource
def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@st.cache_resource
def get_voyage_client() -> voyageai.Client:
    return voyageai.Client()


@st.cache_resource
def get_cached_chroma_collection():
    return get_chroma_collection()


def render_extras(result: dict) -> None:
    """Visualization + debug expander for an already-displayed answer."""
    if result["path"] == "sql" and result.get("raw") and result["raw"]["rows"]:
        columns, rows = result["raw"]["columns"], result["raw"]["rows"]
        df = pd.DataFrame(rows, columns=columns)
        if "total_count" in df.columns:
            # It's the same scalar repeated on every row (a subquery total for the
            # "showing N of X total" headline) -- useful for the synthesis prompt,
            # not for display, where it'd just be a redundant identical column.
            df = df.drop(columns=["total_count"])
        if result.get("is_chart"):
            chart_rows, footnotes = split_base_status_rows(rows)
            render_bar_chart(chart_rows)
            for label, value in footnotes:
                note = EXPIRATION_NOT_STATED_NOTE if str(label).strip().lower() == "expiration not stated" else RENEWAL_UNCERTAIN_NOTE
                st.caption(note.format(n=value))
        elif len(rows) > 1:
            # Spec's literal rule only requires a table for >3 rows (that threshold
            # instead controls headline-vs-full-prose in the synthesis prompt, see
            # should_visualize() in query_router.py). Showing a table for any
            # multi-row list is a superset of that rule, not a violation of it, and
            # matches the "expiring in 90 days" (2-row) test case's expectation.
            df.index = range(1, len(df) + 1)
            st.dataframe(df)

    if result["path"] in ("sql", "vector"):
        with st.expander(f"Routing details (path: {result['path']})"):
            st.write(f"**Reasoning:** {result['reasoning']}")
            if result["path"] == "sql":
                st.code(result.get("sql", ""), language="sql")
                if result.get("raw"):
                    debug_df = pd.DataFrame(
                        [dict(zip(result["raw"]["columns"], row)) for row in result["raw"]["rows"]]
                    )
                    debug_df.index = range(1, len(debug_df) + 1)
                    st.dataframe(debug_df)
            else:
                st.write(f"**Search query:** {result.get('search_query', '')}")
                if result.get("raw"):
                    for chunk in result["raw"]:
                        meta = chunk["metadata"]
                        st.caption(
                            f"{meta['filename']} — "
                            f"{meta['section_title'] or meta['section_number'] or 'n/a'}, "
                            f"p.{meta['page_number']} ({meta['chunk_type']})"
                        )
            if dev_mode and result.get("timings"):
                st.write("**Step timings (s):**", {k: round(v, 3) for k, v in result["timings"].items()})
    elif dev_mode and result.get("timings"):
        with st.expander("Step timings"):
            st.write({k: round(v, 3) for k, v in result["timings"].items()})


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("debug"):
            render_extras(msg["debug"])

question = st.chat_input("Ask a question about the Lake County contracts...")
if not question and "pending_question" in st.session_state:
    # Set by a sidebar "Try asking:" button click (see SUGGESTED_QUERIES above),
    # which can't write directly into st.chat_input's own value -- routing it
    # through session_state + a rerun is the standard pattern for triggering a
    # chat submission from something other than the input box itself.
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing..."):
            result = answer_question(
                question,
                anthropic_client=get_anthropic_client(),
                voyage_client=get_voyage_client(),
                chroma_collection=get_cached_chroma_collection(),
                stream=True,
            )
        answer_text = st.write_stream(result["answer"])
        result["answer"] = answer_text  # replace the (now-exhausted) generator for storage/replay
        render_extras(result)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "debug": result,
    })
