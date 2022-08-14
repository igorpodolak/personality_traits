import pandas as pd


def main():
    per57 = pd.read_csv("personality_57.csv")
    # per_xlsx = pd.read_excel("NEO_przeliczone.xlsx")
    # per_xlsx = pd.read_excel("NEO_recalculated.xlsx")
    per_recalc = pd.read_csv("NEO_przeliczone.csv")
    # per_recalc = per_recalc.astype({'REC N': 'float64', 'REC E': 'int64', 'REC O': 'int64', 'REC U': 'int64',
    #                                 'REC S': 'int64'})
    # df = per57.copy()
    df = pd.merge(per57, per_recalc, on="hash", how="left")
    df.to_csv("personality_57_rec.csv", index=False)
    pass


if __name__ == "__main__":
    main()