def max_value(rows):
  return max(rows)

def filter_metric(rows, metric):
  return [row for row in rows if row["metric"] == metric]
