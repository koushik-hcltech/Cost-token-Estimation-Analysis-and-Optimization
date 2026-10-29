"""Notebook display helpers (plumbing): render measured results prominently."""


def show_result(label, cost_per_query, quality):
    "Render one measured configuration as a metric card."
    from IPython.display import HTML, display
    display(HTML(
        "<div style='border:1px solid #dadce0; border-left:6px solid #4285F4; "
        "padding:10px 16px; margin:8px 0; font-family:sans-serif; background:#f8f9fa'>"
        f"<div style='color:#5f6368; font-size:0.95em'>{label}</div>"
        f"<span style='font-size:1.7em; font-weight:600'>${cost_per_query:.4f}</span>"
        "<span style='color:#5f6368'> per query</span>"
        "<span style='margin:0 12px; color:#dadce0'>|</span>"
        f"<span style='font-size:1.7em; font-weight:600'>{quality:.2f}</span>"
        "<span style='color:#5f6368'> quality</span></div>"))


def show_verdict(passed, reasons):
    "Render the final-challenge verdict as a PASS/FAIL banner."
    from IPython.display import HTML, display
    color, text = ("#188038", "PASS") if passed else ("#d93025", "FAIL")
    why = "" if passed else (
        "<div style='margin-top:4px; font-size:0.95em'>" + "; ".join(reasons) + "</div>")
    display(HTML(
        f"<div style='border-left:6px solid {color}; background:{color}14; color:{color}; "
        "padding:10px 16px; margin:8px 0; font-family:sans-serif'>"
        f"<span style='font-size:1.5em; font-weight:700'>{text}</span>{why}</div>"))


def show_target(budget, quality_floor):
    "Render the final-challenge objective as actual values."
    from IPython.display import HTML, display
    display(HTML(
        "<div style='border:1px solid #dadce0; border-left:6px solid #F29900; "
        "padding:12px 16px; margin:8px 0; font-family:sans-serif; background:#fffdf5'>"
        "<div style='color:#5f6368; font-size:0.95em'>YOUR FINAL CHALLENGE TARGET (both must hold)</div>"
        f"<span style='font-size:1.7em; font-weight:700'>&le; ${budget:.4f}</span>"
        "<span style='color:#5f6368'> per query</span>"
        "<span style='margin:0 12px; color:#dadce0'>|</span>"
        f"<span style='font-size:1.7em; font-weight:700'>&ge; {quality_floor:.2f}</span>"
        "<span style='color:#5f6368'> quality</span></div>"))
