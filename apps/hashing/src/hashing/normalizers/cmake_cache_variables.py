def normalize_cmake(lines: list[str]) -> set:
  """
  Method:
  - From a list of strings, extract "-D", clean, strip and stack them.

  Example 1:
  - Input: ["cmake" "-S", ".", "-B", "build-cuda", "-DCMAKE_BUILD_TYPE=Release"]
  - Output: { "CMAKE_BUILD_TYPE=Release" }

  Example 2:
  - Input: ["cmake", "-DCMAKE_BUILD_TYPE=Debug", "-DTPL_ENABLE_MPI=ON"]
  - Output: { "CMAKE_BUILD_TYPE=Debug", "TPL_ENABLE_MPI=ON" }
  """
  tokens = set()

  for line in lines:
    # clean
    line = line.strip().rstrip("\\").strip()

    # get persistent variable
    if line.startswith("-D"):
      keyval = line[2:]
      key, value = keyval.split("=", 1)
      tokens.add(f"{key.strip()}={value.strip()}")

  return tokens