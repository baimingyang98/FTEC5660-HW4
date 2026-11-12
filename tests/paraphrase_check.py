"""Questions in other words than public_test, on the public corpus: do the rules generalise?

    python tests/paraphrase_check.py

No API calls: it checks parse_query (Task 2.2), and whether retrieve returns the evidence (answerable
questions) or [] (unanswerable ones). Each case: question, expected companies, fiscal_year, doc_type,
and the evidence [(document, text)] a retrieved chunk must contain (one list per fact); None means the
question only has to get chunks (or [] if it is unanswerable).
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
import hw4

NR25, NR24, BE_CALL, SR_RR, OB24 = "doc_038.html", "doc_028.html", "doc_006.html", "doc_010.html", "doc_001.html"
CASES = [
    ("How many people did Northgate Robotics employ at the end of FY2025?", ["Northgate Robotics"], 2025, "annual_report", [[(NR25, "9,104")]]),
    ("What was the headcount of Northgate Robotics at 31 December 2024?", ["Northgate Robotics"], 2024, "annual_report", [[(NR25, "8,073"), (NR24, "8,073")]]),
    ("Who audited Northgate Robotics' consolidated financial statements for FY2025?", ["Northgate Robotics"], 2025, "annual_report", [[(NR25, "Whitlock Byrne")]]),
    ("What is the ticker symbol of Orchid Bay Hotels?", ["Orchid Bay Hotels"], None, "annual_report", [[(OB24, "1876")]]),
    ("Where is Orchid Bay Hotels headquartered?", ["Orchid Bay Hotels"], None, "annual_report", [[(OB24, "Hong Kong")]]),
    ("What diluted EPS did Northgate Robotics report for FY2025, in USD?", ["Northgate Robotics"], 2025, "annual_report", [[(NR25, "2.25")]]),
    ("What 12-month price target has Meridian Equity Research set for Sablewood Retail?", ["Sablewood Retail"], None, "research_report", [[(SR_RR, "66.36")]]),
    ("What rating did Meridian Equity Research assign to Sablewood Retail?", ["Sablewood Retail"], None, "research_report", [[(SR_RR, "HOLD")]]),
    ("What was Meridian Equity Research's previous FY2026 revenue estimate for Sablewood Retail, in USD millions?", ["Sablewood Retail"], 2026, "research_report", [[(SR_RR, "2,476.9")]]),
    ("What FY2027 revenue does the broker project for Sablewood Retail, in USD millions?", ["Sablewood Retail"], 2027, "research_report", [[(SR_RR, "2,862.6")]]),
    ("How much revenue did Northgate Robotics' Collaborative Robots segment generate in 2025?", ["Northgate Robotics"], 2025, "annual_report", [[(NR25, "1,139.2")]]),
    ("What was Northgate Robotics' revenue from the Americas in FY2025, in USD millions?", ["Northgate Robotics"], 2025, "annual_report", [[(NR25, "1,071.4")]]),
    ("What size of share buyback program did Brightwater Energy announce on its fourth-quarter earnings call, in USD millions?", ["Brightwater Energy"], None, "earnings_call", [[(BE_CALL, "200")]]),
    ("According to its earnings call, what revenue growth does Brightwater Energy expect for fiscal 2026, in percent?", ["Brightwater Energy"], 2026, "earnings_call", [[(BE_CALL, "12.5%")]]),
    ("How large a share repurchase programme did Brightwater Energy announce, in USD millions?", ["Brightwater Energy"], None, "annual_report", [[(BE_CALL, "200")]]),
    ("Who was Northgate Robotics' CFO before Thomas Kaur?", ["Northgate Robotics"], None, "annual_report", [[(NR25, "Nadia Moreau")]]),
    ("What was the value of Northgate Robotics' inventory at 31 December 2025, in USD millions?", ["Northgate Robotics"], 2025, "annual_report", [[(NR25, "506.3")]]),
    ("What was Northgate Robotics' total shareholders' equity at the end of FY2025, in USD millions?", ["Northgate Robotics"], 2025, "annual_report", [[(NR25, "1,197.5")]]),
    ("How much did Northgate Robotics spend on R&D in FY2025, in USD millions?", ["Northgate Robotics"], 2025, "annual_report", [[(NR25, "(182.6)")]]),
    ("What was Northgate Robotics' capex in FY2025, in USD millions?", ["Northgate Robotics"], 2025, "annual_report", None),
    ("What was Northgate Robotics' depreciation and amortization in FY2025, in USD millions?", ["Northgate Robotics"], 2025, "annual_report", None),
    ("Which of Northgate Robotics and Northgate Payments had the higher operating income in FY2025?", ["Northgate Robotics", "Northgate Payments"], 2025, "annual_report", "both"),
    ("Between Sablewood Retail and Lumen Harbor Foods, which company reported more revenue in FY2024?", ["Sablewood Retail", "Lumen Harbor Foods"], 2024, "annual_report", "both"),
    # unanswerable
    ("What was Northgate Robotics' revenue in FY2020, in USD millions?", ["Northgate Robotics"], 2020, "annual_report", None),
    ("How much did Northgate Robotics spend on employee training in FY2025, in USD millions?", ["Northgate Robotics"], 2025, "annual_report", None),
    ("What were Sablewood Retail Group's total assets at 31 December 2025, in USD millions?", ["Sablewood Retail Group"], 2025, "annual_report", None),
    ("How much cash did Pemberton Freight hold at 31 December 2025, in USD millions?", ["Pemberton Freight"], 2025, "annual_report", None),
    ("What FY2030 revenue does Meridian Equity Research forecast for Sablewood Retail, in USD millions?", ["Sablewood Retail"], 2030, "research_report", None),
    ("What was Orchid Bay Hotels' greenhouse gas emissions in FY2025, in tonnes?", ["Orchid Bay Hotels"], 2025, "annual_report", None),
]
UNANSWERABLE = {"FY2020", "employee training", "Sablewood Retail Group", "Pemberton Freight", "FY2030", "greenhouse"}

corpus = hw4.load_corpus(repo / "public_test" / "corpus")
chunks = [c for f in sorted(corpus) for c in hw4.chunk_document(f, corpus[f])]
index = hw4.build_index(chunks)
problems = 0
for question, companies, year, doc_type, evidence in CASES:
    parsed = hw4.parse_query(question)
    gold = {"query": {"companies": companies, "fiscal_year": year, "doc_type": doc_type}}
    score = hw4.parse_score(parsed, gold)
    got = hw4.retrieve(index, question)
    unanswerable = any(u in question for u in UNANSWERABLE)
    if unanswerable:
        ok = not got
    elif evidence == "both":
        ok = {hw4.company_key(c.metadata["company"]) for c in got} == {hw4.company_key(c) for c in companies}
    elif evidence is None:
        ok = bool(got)
    else:
        ok = hw4.retrieval_hit(got, {"evidence": [[list(e) for e in grp] for grp in evidence]})
    if score < 1 or not ok:
        problems += 1
        print(f"parse={score:.2f} retrieval={'ok' if ok else 'WRONG'} | {question}\n    parsed={parsed}")
        print(f"    got={[(c.metadata['source'][4:7], c.metadata['section'][-35:]) for c in got]}")
    elif evidence is None and not unanswerable:
        print(f"(check by eye) {question}\n    got={[(c.metadata['source'][4:7], c.metadata['section'][-35:]) for c in got]}")
print(f"\n{len(CASES) - problems}/{len(CASES)} cases right")
