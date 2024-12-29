import pandas as pd
import matplotlib.pyplot as plt


def split_vis(df, output_type="GRAPH"):

    # Grouping by both columns and counting occurrences
    grouped_counts = df.groupby(["fold", "class_name"]).size().unstack(fill_value=0)

    if output_type == "TABLE":
        return grouped_counts

    # Plotting stacked bar chart with Seaborn
    ax = grouped_counts.plot(kind="bar", stacked=True, figsize=(8, 6))

    # Adding count labels on each bar segment
    for i in range(len(grouped_counts)):
        for j in range(len(grouped_counts.columns)):
            plt.text(
                i,
                grouped_counts.iloc[i, : j + 1].sum() - grouped_counts.iloc[i, j] / 2,
                str(grouped_counts.iloc[i, j]),
                ha="center",
                va="center",
            )

    # Adding labels and title
    plt.title("Stacked Bar Chart of Counts by fold and class_idx")
    plt.xlabel("fold")
    plt.ylabel("Count")
    plt.xticks(rotation=0)

    # Showing the plot
    plt.legend(title="Class_idx")
    plt.show()
