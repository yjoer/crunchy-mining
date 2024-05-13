import altair as alt
import pandas as pd


def plot_top_state_by_loan(df: pd.DataFrame):
    df_states = pd.read_csv("data/states.csv")
    df_states.set_index("Abbreviations", inplace=True)

    entries_by_state = df["State"].value_counts().to_frame()
    entries_by_state = entries_by_state.join(df_states, how="left")
    top_ten_state_by_entries = entries_by_state[:10]
    lst_ten_state_by_entries = entries_by_state[-10:]

    base_top = alt.Chart(
        data=top_ten_state_by_entries,
        title="Top Ten States Leading in Loan Applications",
    ).encode(
        x=alt.X("count:Q", title="Count", scale=alt.Scale(domainMax=155000)),
        y=alt.Y("Name:N", title="States").sort("-x"),
        color=alt.Color("Name:N", legend=None, sort="-x"),
        # text=alt.Text("count", format=","),
        tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("count:Q", title="Count")],
    )

    chart_top = base_top.mark_bar() + base_top.mark_text(align="left", dx=2)

    base_lst = alt.Chart(
        data=lst_ten_state_by_entries,
        title="Top Ten States Trailing in Loan Applications",
    ).encode(
        x=alt.X("count:Q", title="Count", scale=alt.Scale(domainMax=6375)),
        y=alt.Y("Name:N", title="States").sort("x"),
        color=alt.Color("Name:N", legend=None, sort="x"),
        # text=alt.Text("count", format=","),
        tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("count:Q", title="Count")],
    )

    chart_lst = base_lst.mark_bar() + base_lst.mark_text(align="left", dx=2)

    return chart_top, chart_lst


def plot_loan_default_rate_by_state(df: pd.DataFrame):
    df_states = pd.read_csv("data/states.csv")
    df_states.set_index("Abbreviations", inplace=True)

    default_by_state = df[df["MIS_Status"] == "CHGOFF"]["State"].value_counts()
    entries_by_state = df["State"].value_counts()
    default_rate_by_states = (default_by_state / entries_by_state).to_frame()

    default_rate_by_states = default_rate_by_states.join(df_states, how="left")

    default_rate_by_states["Name"] = default_rate_by_states["Name"].str.title()
    default_rate_by_states.replace(
        to_replace={"District Of Columbia": "District of Columbia"},
        inplace=True,
    )

    states = alt.topo_feature(
        url="https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json",
        feature="states",
    )

    chart = (
        alt.Chart(states, title="Loan Default Rate by States")
        .mark_geoshape(stroke="white")
        .transform_lookup(
            lookup="properties.name",
            from_=alt.LookupData(default_rate_by_states, "Name", ["count"]),
        )
        .encode(
            color=alt.Color("count:Q", legend=alt.Legend(title="Default Rate")),
            tooltip=[
                alt.Tooltip("properties.name:N", title="State"),
                alt.Tooltip("count:Q", title="Rate"),
            ],
        )
        .project(type="albersUsa")
    )

    return chart


def plot_approved_amount_by_status(df: pd.DataFrame):
    chart = (
        alt.Chart(df[["GrAppv", "MIS_Status"]], title="Approved Loan Amount by Status")
        .mark_bar(binSpacing=0)
        .encode(
            x=alt.X("GrAppv:Q", title="Approved Loan Amount", bin=alt.Bin(maxbins=100)),
            y=alt.Y("count()").scale(type="log"),
            color="MIS_Status:N",
        )
    )

    return chart
