"""Bilinear interpolation functions from VERTCON."""

def interpolate(x: float, y: float, q11: float, q12: float, q21: float, q22: float) -> float:
    """
    Perform bilinear interpolation.
    
    Args:
        x: x coordinate (0-1)
        y: y coordinate (0-1)
        q11: value at (0,0)
        q12: value at (1,0)
        q21: value at (0,1)
        q22: value at (1,1)
    
    Returns:
        Interpolated value
    """
    return (q11 * (1-x) * (1-y) +
            q21 * (1-x) * y +
            q12 * x * (1-y) +
            q22 * x * y) 