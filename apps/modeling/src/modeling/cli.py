import argparse

def main() -> None:
  print("hello from modeling")

  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()

  print(f"input: {args.input}")
  print(f"output: {args.output}")
