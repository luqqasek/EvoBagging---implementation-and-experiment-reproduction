import pandas as pd
import json

if __name__ == "__main__":
    df = pd.read_csv(r'evobagging_grid_search.csv')
    idx = df.groupby(["Dataset_name"])['CV_score'].transform(max) == df['CV_score']
    df[idx].drop_duplicates('Dataset_name')
    final_dict = {}
    output = df[idx].drop_duplicates('Dataset_name').drop("CV_score", axis=1)
    for index, row in output.iterrows():
        params = {"G": row["G"], "M": row["M"], "MS": row["MS"], "K": row["K"]}
        final_dict[row["Dataset_name"]] = params

    with open("params.json", "w") as outfile:
        json.dump(final_dict, outfile)
