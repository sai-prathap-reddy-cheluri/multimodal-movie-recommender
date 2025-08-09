import gradio as gr
from datetime import date, datetime

from dateutil.relativedelta import relativedelta
from src.utils.download_utils import download_movie_data


def ensure_datetime(dt):
    """Convert date/datetime/other to datetime object."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day)
    if isinstance(dt, float) or isinstance(dt, int):
        try:
            return datetime.fromtimestamp(dt)
        except Exception:
            pass
    if isinstance(dt, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(dt, fmt)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(dt)
        except Exception:
            pass
    raise ValueError(f"Unrecognized date format: {dt}")

def to_datetime(d, start=True):
    """Convert a date or datetime to datetime at 00:00 (start) or 23:59 (end)."""
    d = ensure_datetime(d)
    if d is None:
        return None
    return d.replace(hour=0 if start else 23, minute=0 if start else 59, second=0 if start else 59)


def calculate_dates(preset):
    """Return (from_date, to_date) datetimes for a preset."""
    today = date.today()
    if preset == "Last 3 months":
        from_date = to_datetime(today - relativedelta(months=3), start=True)
        to_date = to_datetime(today, start=False)
        return from_date, to_date
    elif preset == "Last 6 months":
        from_date = to_datetime(today - relativedelta(months=6), start=True)
        to_date = to_datetime(today, start=False)
        return from_date, to_date
    elif preset == "Last 1 year":
        from_date = to_datetime(today - relativedelta(years=1), start=True)
        to_date = to_datetime(today, start=False)
        return from_date, to_date
    elif preset == "Current Year":
        from_date = to_datetime(date(today.year, 1, 1), start=True)
        to_date = to_datetime(today, start=False)
        return from_date, to_date
    else:  # Custom
        from_date = to_datetime(today, start=True)
        to_date = to_datetime(today, start=False)
        return from_date, to_date


def convert_date_format(dt, end=False):
    """Format datetime to YYYY-MM-DD (set time to end-of-day if end=True)."""
    dt = ensure_datetime(dt)
    if dt is None:
        return ""
    if end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return dt.strftime("%Y-%m-%d")


def output_filename(from_date, to_date):
    """Generate CSV filename based on date range."""
    backend_from = convert_date_format(from_date)
    backend_to = convert_date_format(to_date)
    filename = f"movies_{backend_from}_{backend_to}.csv"
    return filename, backend_from, backend_to

def run_downloader(from_date, to_date, include_adult, concurrency):

    filename, backend_from, backend_to = output_filename(from_date, to_date)
    path = download_movie_data(from_date=backend_from, to_date=backend_to, filename=filename,
                               include_adult=include_adult, region = None,max_pages=500, day_threshold=1,
                               detail_concurrency=int(concurrency))
    if path:
        return (
        f"<div class='download-result'><b>✅ Downloaded!</b><br>"
            f"<b>Saved to:</b> <span>{path}</span><br>"
            f"<b>Time range:</b> <span>{backend_from} <b>to</b> {backend_to}</span><br>"
            f"<b>Adult content included?:</b> <span>{'Yes' if include_adult else 'No'}</span>"
            f"</div>"
        )
    else:
        return "<div class='download-result' style='color:#ff7675'><b>No movies found for your selection.</b></div>"


def handle_time_preset(preset):
    if preset == "Custom":
        # Show calendar pickers for custom
        return gr.update(visible=True, value=None), gr.update(visible=True, value=None)
    else:
        fd, td = calculate_dates(preset)
        return gr.update(visible=False, value=fd), gr.update(visible=False, value=td)

def autofill_to_date(from_date_val, to_date_val):
    # If To Date is empty and From Date is set, set To Date to today's date
    if to_date_val is None and from_date_val is not None:
        # Today's date as datetime with 23:59:59 (end of day)
        today = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
        return today
    return to_date_val

custom_css = """
.download-result {
    background: linear-gradient(90deg,#23243a,#38f9d7);
    border-radius: 18px;
    color: #fff;
    font-size: 1.0em;
    padding: 12px 16px;
    margin-top: 16px;
}
#adult-checkbox {
    margin-top: 14px; 
}
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as demo:
    gr.HTML(
        """
        <div id='custom-header'>
        <h1>🎬 Movie Dataset Downloader</h1>
        <p>Select a preset or choose your own dates, languages, and adult filter. Dataset will be saved and ready for analysis!</p>
        </div>
        """
    )

    # Time range selection
    time_preset = gr.Dropdown(
        label = "Select Time Range",
        choices = ["Last 3 months", "Last 6 months", "Last 1 year", "Current Year", "Custom"],
        value = "Last 3 months"
    )

    with gr.Row():
        from_date = gr.DateTime(
            label = "From Date",
            visible = True,
            interactive = True,
            value = calculate_dates("Last 3 months")[0]
        )
        to_date = gr.DateTime(
            label = "To Date",
            visible = True,
            interactive = True,
            value = calculate_dates("Last 3 months")[1]
        )

    with gr.Row():
        include_adult = gr.Checkbox(
            label="Include adult (18+) movies",
            value=False,
            elem_id="adult-checkbox",
            scale=1
        )
        concurrency = gr.Slider(
            5, 50,
            value=20,
            step=5,
            label="Concurrency (number of parallel detail requests)",
            scale=3
        )

    time_preset.change(
        handle_time_preset,
        inputs = time_preset,
        outputs = [from_date, to_date]
    )

    from_date.change(
        autofill_to_date,
        inputs=[from_date, to_date],
        outputs=to_date
    )
    download_button = gr.Button("Download")
    result = gr.HTML()

    download_button.click(
        run_downloader,
        inputs=[from_date, to_date, include_adult, concurrency],
        outputs=result,
    )


if __name__ == "__main__":
    demo.launch()
