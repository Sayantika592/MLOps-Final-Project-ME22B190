"""
Sample Data Generator
=====================
Creates a small synthetic dataset for testing the pipeline
when the full Kaggle dataset is not yet available.

Usage:
    python scripts/generate_sample_data.py
"""

import os
import pandas as pd
import numpy as np

REAL_TITLES = [
    "Federal Reserve raises interest rates by quarter point",
    "Stock market closes higher on strong earnings reports",
    "New study links exercise to improved mental health outcomes",
    "Congress passes bipartisan infrastructure bill",
    "Scientists develop new vaccine for respiratory virus",
    "Local school district announces new curriculum changes",
    "Tech company reports record quarterly revenue growth",
    "City council approves new public transportation plan",
    "University researchers publish findings on climate patterns",
    "Hospital implements new patient safety protocols",
    "Economic growth exceeds analyst expectations this quarter",
    "Government announces new environmental protection regulations",
    "Medical researchers identify new treatment for chronic pain",
    "International trade agreement reaches final negotiations",
    "Census data reveals shifting population demographics",
]

FAKE_TITLES = [
    "SHOCKING: Secret government mind control program exposed",
    "You won't believe what this celebrity did to lose weight overnight",
    "Scientists confirm the earth is actually flat new evidence shows",
    "BREAKING: Aliens found living among us government coverup revealed",
    "This one weird trick will make you a millionaire in just one week",
    "EXPOSED: All vaccines are actually poison designed to control population",
    "Celebrity secretly a lizard person according to insider sources",
    "Moon landing was filmed in a Hollywood studio declassified documents show",
    "URGENT: Secret cure for all diseases suppressed by big pharma",
    "Government plans to ban all private vehicles shocking leaked documents",
    "Time traveler from 2050 warns of impending global catastrophe",
    "Underground city discovered beneath major metropolitan area",
    "Mysterious signals from deep space are actually alien communications",
    "BREAKING: Major world leader is actually a clone replacement",
    "Scientists accidentally open portal to another dimension",
]

REAL_TEXTS = [
    "According to the Federal Reserve's latest report, the central bank has decided to increase the benchmark interest rate by 25 basis points. This decision comes after months of deliberation and reflects the committee's assessment of current economic conditions including employment data and inflation metrics.",
    "Wall Street experienced a positive trading session today as major indices closed higher following better-than-expected corporate earnings reports. The S&P 500 gained 1.2% while the Dow Jones Industrial Average rose by approximately 300 points.",
    "A comprehensive study published in the Journal of Clinical Psychology has found significant correlations between regular physical exercise and improvements in mental health outcomes. The research, conducted over five years with 10,000 participants, demonstrates measurable benefits.",
    "In a bipartisan vote, Congress passed the long-awaited infrastructure bill allocating $500 billion for road repairs, bridge maintenance, and broadband expansion. The legislation received support from both sides of the aisle.",
    "Researchers at the National Institutes of Health have completed Phase 3 trials for a new vaccine targeting a common respiratory virus. The results show 89% efficacy with minimal side effects reported among the study participants.",
]

FAKE_TEXTS = [
    "SHOCKING REVELATION: Top secret documents leaked from a classified government facility prove that world leaders have been using advanced mind control technology on the general population for decades. Anonymous sources confirm the program has been active since the 1960s and affects millions.",
    "Doctors are FURIOUS about this one simple trick that helps people lose 50 pounds in just one week without any exercise or diet changes. Big pharmaceutical companies have been trying to suppress this information for years because it would destroy their billion dollar weight loss industry.",
    "A group of independent researchers have finally proven beyond any doubt that the earth is completely flat. Their groundbreaking experiments conducted from various locations around the globe definitively show that all previous scientific evidence was fabricated by space agencies worldwide.",
    "Multiple credible witnesses have come forward with undeniable evidence that extraterrestrial beings have been living among the human population for at least the past century. Government officials at the highest levels have been aware of this fact and have actively covered it up.",
    "A secret miracle cure that eliminates all known diseases has been discovered by a small team of rogue scientists. However powerful pharmaceutical corporations and corrupt government agencies have conspired to suppress this discovery because it would eliminate their enormous profits.",
]


def generate_sample_data(output_path: str = "data/raw/news.csv", n_samples: int = 500):
    """Generate a synthetic fake news dataset."""
    np.random.seed(42)

    records = []
    for i in range(n_samples):
        if i % 2 == 0:  # Real news
            title = np.random.choice(REAL_TITLES)
            text = np.random.choice(REAL_TEXTS)
            # Add some variation
            text = text + " " + " ".join(np.random.choice(
                ["The data supports this conclusion.", "Officials confirmed the report.",
                 "Analysts expect continued trends.", "The findings were peer reviewed.",
                 "Multiple sources corroborated the information."],
                size=np.random.randint(1, 4),
            ))
            label = 0
        else:  # Fake news
            title = np.random.choice(FAKE_TITLES)
            text = np.random.choice(FAKE_TEXTS)
            text = text + " " + " ".join(np.random.choice(
                ["SHARE THIS BEFORE THEY DELETE IT!", "The mainstream media won't tell you this!",
                 "Wake up people the truth is out there!", "They don't want you to know!",
                 "This is being censored everywhere!"],
                size=np.random.randint(1, 4),
            ))
            label = 1

        author = np.random.choice(
            ["John Smith", "Jane Doe", "Anonymous", "Staff Reporter",
             "Editorial Board", "Sarah Johnson", "Mike Chen", ""]
        )

        records.append({
            "id": i,
            "title": title,
            "author": author,
            "text": text,
            "label": label,
        })

    df = pd.DataFrame(records)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} samples → {output_path}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    generate_sample_data()
