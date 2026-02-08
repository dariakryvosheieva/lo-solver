from rosetta import format_solution_rosetta

def solve(problem, type="rosetta"):
    """Route the problem to the appropriate solver."""
    if type == "rosetta":
        format_solution_rosetta(problem)
    else:
        raise ValueError(f"Unknown problem type: {type}")