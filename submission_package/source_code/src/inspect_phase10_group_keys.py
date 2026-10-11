import pandas as pd

path = "results/tables/phase10o_human_validation/phase10o_example_rating_sheet.csv"
df = pd.read_csv(path)

print("Model names:")
print(df["model"].value_counts(dropna=False).to_string())

print("\nScience topic groups:")
print(
    df.loc[df["community"].eq("science"), ["model", "topic"]]
      .drop_duplicates()
      .sort_values(["model", "topic"])
      .to_string(index=False)
)
