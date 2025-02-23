import sys
import json
import re
import pandas as pd

def extract_json_blocks(file_path):
    """ Extracts JSON blocks containing 'entryPoints', 'start', and 'end' fields. """
    json_blocks = []
    buffer = ""
    inside_json = False  # Flag to track JSON parsing state

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            # If line contains "entryPoints", it's the start of a JSON object
            if '"entryPoints"' in line:
                buffer = line
                inside_json = True
            elif inside_json:
                buffer += line  # Append to the JSON buffer

            # If JSON block is fully formed (ends with '}'), parse it
            if inside_json and buffer.endswith("}"):
                try:
                    json_data = json.loads(buffer)
                    if "entryPoints" in json_data and "start" in json_data and "end" in json_data:
                        json_blocks.append(json_data)
                except json.JSONDecodeError:
                    pass  # Ignore malformed JSON
                buffer = ""
                inside_json = False  # Reset flag

    return json_blocks

def process_kernel_data(json_blocks):
    """ Process extracted JSON data to compute execution times in milliseconds. """
    kernel_data = []

    for entry in json_blocks:
        try:
            kernel_name = entry["entryPoints"][0]
            start_time = int(entry["start"])
            end_time = int(entry["end"])
            exec_time_ms = (end_time - start_time) / 1e6  # Convert ns to ms
            
            kernel_data.append({
                "Kernel Name": kernel_name,
                "Execution Time (ms)": exec_time_ms
            })
        except (KeyError, ValueError):
            continue  # Skip incomplete/malformed entries

    return kernel_data

def generate_statistics(kernel_data):
    """ Generate execution statistics for kernels. """
    df = pd.DataFrame(kernel_data)

    if df.empty:
        print("No valid kernel execution data found. Using alternative approach...")
        sys.exit(1)

    # Compute kernel execution count, total time, average time, and peak execution time
    stats = df.groupby("Kernel Name").agg(
        Iterations=("Execution Time (ms)", "count"),
        Total_Time_ms=("Execution Time (ms)", "sum"),
        Avg_Time_ms=("Execution Time (ms)", "mean"),
        Peak_Time_ms=("Execution Time (ms)", "max")  # Peak execution time
    ).reset_index()

    # Sort by Total Execution Time (most time-consuming kernels)
    stats["% of Total"] = (stats["Total_Time_ms"] / stats["Total_Time_ms"].sum()) * 100
    stats = stats.sort_values(by="Total_Time_ms", ascending=False)
    stats["Cumulative %"] = stats["% of Total"].cumsum()

    # Format percentage columns
    stats["% of Total"] = stats["% of Total"].map(lambda x: f"{x:.2f}%")
    stats["Cumulative %"] = stats["Cumulative %"].map(lambda x: f"{x:.2f}%")

    # Get the top 10 unique most time-consuming kernels with peak time
    top_10_unique_kernels = stats[["Kernel Name", "Peak_Time_ms"]].head(10)

    return top_10_unique_kernels, stats

def print_table(title, df, columns):
    """ Print a properly formatted table with left-aligned kernel names and '|' separator. """
    
    # Determine the max length of kernel names dynamically
    max_kernel_length = max(df["Kernel Name"].apply(len).max(), len("Kernel Name")) + 2

    # Define column widths dynamically
    column_widths = {
        "Kernel Name": max_kernel_length,
        "Total_Time_ms": 15,
        "Iterations": 10,
        "Avg_Time_ms": 15,
        "Peak_Time_ms": 15,
        "% of Total": 12,
        "Cumulative %": 12
    }

    print("\n" + title)
    print("-" * len(title))

    # Print header
    header = " | ".join(f"{col:<{column_widths[col]}}" for col in columns)
    print(header)
    print("-" * len(header))

    # Print data rows
    for _, row in df.iterrows():
        line = " | ".join(
            f"{row[col]:<{column_widths[col]}.2f}" if "Time" in col else f"{row[col]:<{column_widths[col]}}"
            for col in columns
        )
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python log_analyzer.py <log_file.txt>")
        sys.exit(1)

    log_file_path = sys.argv[1]
    json_blocks = extract_json_blocks(log_file_path)

    if not json_blocks:
        # Try an alternative regex-based extraction if JSON parsing fails
        with open(log_file_path, 'r', encoding='utf-8') as f:
            raw_data = f.read()

        matches = re.findall(r'\{[^{}]*"entryPoints"[^{}]*\}', raw_data)
        json_blocks = [json.loads(m) for m in matches if "start" in m and "end" in m]

        if not json_blocks:
            print("No kernel data found. Check file format.")
            sys.exit(1)

    top_10_kernels, complete_stats = generate_statistics(process_kernel_data(json_blocks))

    # Print the first table (Top 10 unique time-consuming kernels with peak execution time)
    print_table(
        "Top 10 Most Time-Consuming Kernels",
        top_10_kernels,
        ["Kernel Name", "Peak_Time_ms"]
    )

    # Print the second table (Complete statistics)
    print_table(
        "Kernel Execution Statistics (Sorted by Total Time)",
        complete_stats,
        ["Kernel Name", "Total_Time_ms", "Iterations", "Avg_Time_ms", "Peak_Time_ms", "% of Total", "Cumulative %"]
    )

