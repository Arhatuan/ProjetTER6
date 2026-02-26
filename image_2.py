import math
from extendedRLM import image as ImageRLM


def distance_descriptors(objects, x, y, lines) -> tuple[list[float], list[float]]:
    """Compute the distance descriptors : for each line in the RLM model, 
    the distance between the center point and the intersection points between the line and each of the two shapes.

    Args:
        objects: objects to compute data from.
        x: x coordinate of the middle point.
        y: y coordinate of the middle point.
        lines: half-lines from the middle point.

    Returns:
        dist1,dist2 (tuple[ list[float], list[float] ]): two lists of the size of the number of half-lines
    """
    dist1 = []
    dist2 = []
    
    # 'line' consists of the discrete points from the center (x,y) in a direction, up to the length of a specific radius
    for line in lines:
        intersectionPointA = None
        intersectionPointB = None

        for pt in line:

            if (intersectionPointA is None) or (intersectionPointB is None):
                if intersectionPointA is None and 0 <= pt[0] < len(objects[1]) and 0 <= pt[1] < len(objects[1][0]) and objects[1][pt[0]][pt[1]]:
                    intersectionPointA = pt
                elif intersectionPointB is None and 0 <= pt[0] < len(objects[0]) and 0 <= pt[1] < len(objects[0][0]) and objects[0][pt[0]][pt[1]]:
                    intersectionPointB = pt

                # symetric point, with respect to the central point (so in the other direction)
                pt_symetric = (2*x - pt[0], 2*y - pt[1])

                if intersectionPointA is None and 0 <= pt_symetric[0] < len(objects[1]) and 0 <= pt_symetric[1] < len(objects[1][0]) and objects[1][pt_symetric[0]][pt_symetric[1]]:
                    intersectionPointA = pt
                elif intersectionPointB is None and 0 <= pt_symetric[0] < len(objects[0]) and 0 <= pt_symetric[1] < len(objects[0][0]) and objects[0][pt_symetric[0]][pt_symetric[1]]:
                    intersectionPointB = pt
    
        distanceToObjectA = math.dist((x,y), (intersectionPointA[0], intersectionPointA[1])) if intersectionPointA is not None else 0
        dist1.append(distanceToObjectA)
        distanceToObjectB = math.dist((x,y), (intersectionPointB[0], intersectionPointB[1])) if intersectionPointB is not None else 0
        dist2.append(distanceToObjectB)

    return (dist1, dist2)

def image_processing_v2(imagename, background, step, force_type):
    """
    Compute from an image, the RLM of the first and the second object and forces histogram.
    :param imagename: name of the file of the image.
    :param background: background color of the image.
    :param step: step of the angle needed to compute half-lines and diameters for the RLM and F-histogram.
    :param force_type: type of force to use for the computation (usually 0 or 2).
    :return: 5 histograms of the same size: RLM1, RLM2, F-histogram, DIST1, DIST2.
    """
    objects = ImageRLM.image_segmentation(imagename, background)
    x, y = ImageRLM.center_point(objects)
    lines, diameters = ImageRLM.lines_diameters(objects, x, y, step * math.pi / 180)
    rlm1, rlm2 = ImageRLM.radial_line_model(lines, objects)
    force = ImageRLM.forces(objects, diameters, force_type)
    dist1, dist2 = distance_descriptors(objects, x, y, lines)
    return rlm1, rlm2, force, dist1, dist2
