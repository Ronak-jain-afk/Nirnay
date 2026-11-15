import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infer import NirnayEngine

def main():
    engine = NirnayEngine.from_pretrained("checkpoints/nirnay-0.5b-pilot")

    print("\n--- Decision 1: Snake Navigation (Choice) ---")
    res1 = engine.decide(
        state="Coordinates: x increases right, y increases down, board is 6x6. Snake head at [3,4], food at [5,5].",
        question="Choose a direction to collect food and avoid walls.",
        options=["up", "right", "down"],
        kind="choice",
    )
    print(f"Decision:      {res1.decision}")
    print(f"Probabilities: {res1.probabilities}")
    print(f"Logits:        {res1.logits}")
    print(f"Latency:       {res1.latency_ms:.2f} ms")

    print("\n--- Decision 2: Chemical Process Hazard (Score 0..5) ---")
    res2 = engine.decide(
        state="A grignard reaction quenches with methanol after complete conversion; additionally has an exotherm exceeding cooling capacity.",
        question="Rate process hazard for this scenario on a 0-5 scale.",
        options=["0", "1", "2", "3", "4", "5"],
        kind="score",
    )
    print(f"Decision:      {res2.decision}")
    print(f"Probabilities: {res2.probabilities}")
    print(f"Logits:        {res2.logits}")
    print(f"Latency:       {res2.latency_ms:.2f} ms")

    print("\n--- Decision 3: Document Query Relevance (Noul Binary) ---")
    res3 = engine.decide(
        state="Average annual rainfall in Boston is approximately 43.8 inches, evenly distributed across seasons.",
        question="Is this memory relevant for answering the query? Query: What is the average rainfall in the City of Boston per year?",
        options=["false", "true"],
        kind="noul",
    )
    print(f"Decision:      {res3.decision}")
    print(f"Probabilities: {res3.probabilities}")
    print(f"Logits:        {res3.logits}")
    print(f"Latency:       {res3.latency_ms:.2f} ms")

if __name__ == "__main__":
    main()
