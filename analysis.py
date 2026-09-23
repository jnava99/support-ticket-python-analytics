"""
analysis.py — Support Ticket Analytics (Python/pandas version)

Loads the same support-ticket dataset used in the companion SQL project
(support-ticket-sql-analysis) and reproduces the core analysis in
pandas, plus generates a few charts. Intended as a "same data, different
tool" companion piece — SQL for the queries, Python for analysis + viz.

Run:
    pip install -r requirements.txt
    python analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

PRIORITY_ORDER = ["Urgent", "High", "Medium", "Low"]


def load_data():
    tickets = pd.read_csv(DATA_DIR / "tickets.csv", parse_dates=["created_at", "resolved_at"])
    agents = pd.read_csv(DATA_DIR / "agents.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    return tickets, agents, customers


def add_derived_columns(tickets: pd.DataFrame) -> pd.DataFrame:
    tickets = tickets.copy()
    tickets["resolution_hours"] = (
        tickets["resolved_at"] - tickets["created_at"]
    ).dt.total_seconds() / 3600
    tickets["is_resolved"] = tickets["resolved_at"].notna()
    tickets["breached_sla"] = tickets["is_resolved"] & (
        tickets["resolution_hours"] > tickets["sla_hours"]
    )
    return tickets


def avg_resolution_by_priority(tickets: pd.DataFrame) -> pd.Series:
    resolved = tickets[tickets["is_resolved"]]
    return (
        resolved.groupby("priority")["resolution_hours"]
        .mean()
        .round(1)
        .reindex(PRIORITY_ORDER)
    )


def sla_compliance_by_priority(tickets: pd.DataFrame) -> pd.DataFrame:
    resolved = tickets[tickets["is_resolved"]]
    grouped = resolved.groupby("priority")
    compliance = (
        (1 - grouped["breached_sla"].mean()) * 100
    ).round(1).reindex(PRIORITY_ORDER)
    counts = grouped.size().reindex(PRIORITY_ORDER)
    return pd.DataFrame({"resolved_tickets": counts, "sla_compliance_pct": compliance})


def workload_by_agent(tickets: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    merged = tickets.merge(agents, on="agent_id")
    resolved = merged[merged["is_resolved"]]
    summary = merged.groupby("agent_name").agg(
        total_tickets=("ticket_id", "count"),
        open_tickets=("is_resolved", lambda s: (~s).sum()),
    )
    avg_res = resolved.groupby("agent_name")["resolution_hours"].mean().round(1)
    summary["avg_resolution_hours"] = avg_res
    return summary.sort_values("total_tickets", ascending=False)


def category_breakdown(tickets: pd.DataFrame) -> pd.Series:
    return tickets["category"].value_counts()


def top_customers(tickets: pd.DataFrame, customers: pd.DataFrame, n=5) -> pd.DataFrame:
    merged = tickets.merge(customers, on="customer_id")
    return (
        merged.groupby(["customer_name", "plan_tier"])
        .size()
        .reset_index(name="ticket_count")
        .sort_values("ticket_count", ascending=False)
        .head(n)
    )


def monthly_trend(tickets: pd.DataFrame) -> pd.Series:
    return tickets["created_at"].dt.to_period("M").value_counts().sort_index()


def make_charts(tickets, agents, customers):
    # Chart 1: SLA compliance by priority
    compliance = sla_compliance_by_priority(tickets)
    fig, ax = plt.subplots(figsize=(6, 4))
    compliance["sla_compliance_pct"].plot(kind="bar", ax=ax, color="#4479A1")
    ax.set_ylabel("SLA compliance (%)")
    ax.set_title("SLA Compliance Rate by Priority")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "sla_compliance_by_priority.png", dpi=120)
    plt.close(fig)

    # Chart 2: Ticket volume by category
    cats = category_breakdown(tickets)
    fig, ax = plt.subplots(figsize=(6, 4))
    cats.plot(kind="barh", ax=ax, color="#06AC38")
    ax.set_xlabel("Ticket count")
    ax.set_title("Ticket Volume by Category")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "tickets_by_category.png", dpi=120)
    plt.close(fig)

    # Chart 3: Monthly ticket trend
    trend = monthly_trend(tickets)
    fig, ax = plt.subplots(figsize=(6, 4))
    trend.plot(kind="line", marker="o", ax=ax, color="#E97627")
    ax.set_ylabel("Tickets opened")
    ax.set_title("Monthly Ticket Volume")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "monthly_trend.png", dpi=120)
    plt.close(fig)

    print(f"Saved 3 charts to {CHARTS_DIR}/")


def main():
    tickets, agents, customers = load_data()
    tickets = add_derived_columns(tickets)

    print("=== Average resolution time by priority (hours) ===")
    print(avg_resolution_by_priority(tickets), "\n")

    print("=== SLA compliance by priority ===")
    print(sla_compliance_by_priority(tickets), "\n")

    print("=== Workload by agent ===")
    print(workload_by_agent(tickets, agents), "\n")

    print("=== Ticket volume by category ===")
    print(category_breakdown(tickets), "\n")

    print("=== Top 5 customers by ticket volume ===")
    print(top_customers(tickets, customers).to_string(index=False), "\n")

    print("=== Monthly ticket trend ===")
    print(monthly_trend(tickets), "\n")

    make_charts(tickets, agents, customers)


if __name__ == "__main__":
    main()
