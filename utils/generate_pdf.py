from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import os
import json
import csv
import textwrap


def generate_pdf_report(player_name, report_dir="reports/"):
    json_path = os.path.join(report_dir, f"{player_name}_report.json")
    csv_path = os.path.join(report_dir, f"{player_name}_moves.csv")
    pie_chart_path = os.path.join(report_dir, f"{player_name}_pie_chart.png")
    pdf_path = os.path.join(report_dir, f"{player_name}_report.pdf")

    if not os.path.exists(json_path) or not os.path.exists(csv_path):
        print("[✗] Missing report files.")
        return

    with open(json_path) as f:
        report = json.load(f)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        moves = list(reader)

    worst_moves = sorted(moves, key=lambda x: abs(float(x["CP Loss"])), reverse=True)[:5]

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    margin = 50
    x, y = margin, height - margin

    def new_page():
        nonlocal y
        c.showPage()
        y = height - margin

    def write_wrapped(text, x, y, max_width=90, line_height=15):
        lines = textwrap.wrap(text, width=max_width)
        for line in lines:
            c.drawString(x, y, line)
            y -= line_height
        return y

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.darkblue)
    c.drawString(x, y, f"Chess Game Report: {player_name}")
    c.setFillColor(colors.black)
    y -= 30

    # Game Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Summary:")
    y -= 20

    c.setFont("Helvetica", 12)
    for label in ["accuracy", "total_moves", "blunders", "mistakes", "inaccuracies"]:
        c.drawString(x, y, f"{label.replace('_', ' ').capitalize()}: {report[label]}")
        y -= 18

    y -= 10

    # Suggestions
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Recommendations:")
    y -= 20

    c.setFont("Helvetica", 12)
    for s in report["suggestions"]:
        y = write_wrapped(f"• {s}", x + 10, y)
        y -= 5
        if y < 100:
            new_page()

    y -= 20

    # Pie Chart
    if os.path.exists(pie_chart_path):
        try:
            pie = ImageReader(pie_chart_path)
            c.drawImage(pie, x, y - 280, width=300, height=280)
            y -= 300
        except Exception as e:
            c.drawString(x, y, f"[Image Error] {e}")
            y -= 30
    else:
        c.setFillColor(colors.red)
        c.drawString(x, y, "[!] Pie chart not found.")
        c.setFillColor(colors.black)
        y -= 30

    # Worst Moves
    if y < 150:
        new_page()

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Top 5 Worst Moves (by CP Loss):")
    y -= 20
    c.setFont("Helvetica", 12)

    for move in worst_moves:
        move_line = f"Move {move['Move Number']}: {move['Move']} — CP Loss: {move['CP Loss']} ({move['Classification']})"
        y = write_wrapped(move_line, x, y)
        y -= 5
        if y < 100:
            new_page()

    c.save()
    print(f"[✓] PDF Report saved: {pdf_path}")

