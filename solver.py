from rosetta import format_solution_rosetta

def solve(problem, type="rosetta", debug=False):
    """Route the problem to the appropriate solver."""
    if type == "rosetta":
        format_solution_rosetta(problem, debug=debug)
    elif type == "scrambled_rosetta":
        format_solution_rosetta(problem, scrambled=True, debug=debug)
    else:
        raise ValueError(f"Unknown problem type: {type}")