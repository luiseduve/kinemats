class RotationRepresentations(Enum):
    """
    Enum to access different types of data representations
    """
    Quaternion = "Quaternion"
    EulerDegrees = "Euler"

    def __str__(self):
        return super().value.__str__()

class DistanceMetric(Enum):
    """
    Enum to access different types of data representations
    """
    Correlation = "Quaternion"
    Euclidean = "Euler"
    

    def __str__(self):
        return super().value.__str__()
