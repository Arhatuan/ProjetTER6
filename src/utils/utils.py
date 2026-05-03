import re

from ..descriptors.utils.DescriptorEnum import Descriptor, FORCE_REGEX

def get_unique_force_degrees_and_force_descriptors(unique_descriptors: list[Descriptor | str]) -> tuple[list[float], list[str]]:
    """Return the unique force degrees (2, 0, -1.3...) and force descriptors (f2, f0, f-1.3...) from a list of unique descriptors (no duplicates).

    Returns:
        forceDegrees, forceDescriptors (tuple[list[float], list[str]]): the list of force values (2, 0...) and force descriptors (f2, f0...)
    """
    force_degrees = []
    forces_descriptors = []
    pattern = FORCE_REGEX

    # We expect force descriptors to be of type 'str', 
    #   because there can be an infinite number of possible forces, 
    #   thus force descriptions can't be in the enumeration 'Descriptor'
    # In particular, we expect the forces to be written 'f2', 'f1.3', 'f-0.7'...
    for descriptor in unique_descriptors:
        if type(descriptor) is str:
            match = re.match(pattern, descriptor)
            if match:
                force_degrees.append(float(match.group(1)))
                forces_descriptors.append(descriptor)

    return (force_degrees, forces_descriptors)


def get_key_descriptors_combination(descriptors_combination: list[Descriptor | str]) -> str:
    """From a list representing a descriptors's combination, return its string format, 
    with '+' between the descriptors (like 'rlm+f2+f0')

    Args:
        descriptors_combination (list[Descriptor | str]): the list of descriptors representing a combination

    Returns:
        str: the string version, with '+' between descriptors ; for example 'rlm+f0+f2'
    """
    get_str_descriptor = lambda descriptor: descriptor if type(descriptor) is str else descriptor.value
    return "+".join([get_str_descriptor(descriptor) for descriptor in descriptors_combination])

