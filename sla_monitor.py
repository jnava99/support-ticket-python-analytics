"""
sla_monitor.py — SLA Breach Monitor

A lightweight simulation of the kind of alerting logic behind tools like
PagerDuty or Opsgenie: scan open tickets, flag anything that has already
breached its SLA or is about to, and print an alert feed sorted by urgency.

Since this runs against a static historical sample dataset (not a live
system), it uses SIMULATED_NOW below as its reference "current time"
instead of the real clock. In a live system you'd just use datetime.now().

Run:
    python sla_monitor.py
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# The dataset's most recent timestamp — treated as "now" for this demo.
SIMULATED_NOW = pd.Timestamp("2026-03-20 18:00:00")

AT_RISK_WINDOW_HOURS = 2  # flag tickets within this many hours of breaching


def load_open_tickets() -> pd.DataFrame:
    tickets = pd.read_csv(DATA_DIR / "tickets.csv", parse_dates=["created_at", "resolved_at"])
    agents = pd.read_csv(DATA_DIR / "agents.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")

    open_tickets = tickets[tickets["resolved_at"].isna()].copy()
    open_tickets = open_tickets.merge(agents, on="agent_id").merge(customers, on="customer_id")

    open_tickets["hours_open"] = (
        (SIMULATED_NOW - open_tickets["created_at"]).dt.total_seconds() / 3600
    ).round(1)
    open_tickets["hours_until_breach"] = (
        open_tickets["sla_hours"] - open_tickets["hours_open"]
    ).round(1)

    def classify(hours_until_breach):
        if hours_until_breach < 0:
            return "BREACHED"
        elif hours_until_breach <= AT_RISK_WINDOW_HOURS:
            return "AT RISK"
        return "OK"

    open_tickets["alert_level"] = open_tickets["hours_until_breach"].apply(classify)
    return open_tickets


LEVEL_ICON = {"BREACHED": "🔴", "AT RISK": "🟡", "OK": "🟢"}


def print_alert_feed(open_tickets: pd.DataFrame):
    alerts = open_tickets[open_tickets["alert_level"] != "OK"].sort_values("hours_until_breach")

    total = len(open_tickets)
    n_breached = (open_tickets["alert_level"] == "BREACHED").sum()
    n_at_risk = (open_tickets["alert_level"] == "AT RISK").sum()
    n_ok = total - n_breached - n_at_risk

    print(f"SLA Monitor — {total} open tickets as of {SIMULATED_NOW}")
    print(f"  {n_breached} breached | {n_at_risk} at risk | {n_ok} within SLA\n")

    if alerts.empty:
        print("No active alerts. All open tickets are within SLA.")
        return

    for _, t in alerts.iterrows():
        icon = LEVEL_ICON[t["alert_level"]]
        if t["alert_level"] == "BREACHED":
            time_note = f"overdue by {abs(t['hours_until_breach'])}h"
        else:
            time_note = f"{t['hours_until_breach']}h until SLA breach"

        print(
            f"{icon} {t['alert_level']:<9} | Ticket #{t['ticket_id']} | {t['priority']:<6} "
            f"| {t['customer_name']} | Agent: {t['agent_name']} | {time_note}"
        )


def main():
    open_tickets = load_open_tickets()
    print_alert_feed(open_tickets)


if __name__ == "__main__":
    main()
