import math
import numpy as np
import cv2 as cv
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


def angle_descriptor(objects) -> list[float]:
    """Based of the angles of every (point in object A, point in object B) couples of point,
    compute probabilities of belonging to either "Right", "Left", "Above", "Under" directions.

    Args:
        objects: objects to compute data from.

    Returns:
        probabilities (list[float]): probabilities for "Right", "Left", "Above" and "Under" (in this order)
    """

    # 0) Reduce the image's size so that there aren't too much couples of pixels to compute the angles
    #       (or else it takes seconds to compute, instead of milliseconds)
    objectA = objects[0]
    objectB = objects[1]
    nbCouplesPointsInitial = len(objectA[objectA == True]) * len(objectB[objectB == True])

    if nbCouplesPointsInitial > 10000:
        ratioResize = 1/4
        objectA = cv.resize(objectA.astype("uint8"), dsize=(int(objectA.shape[0] * ratioResize), int(objectA.shape[1] * ratioResize))).astype("bool")
        objectB = cv.resize(objectB.astype("uint8"), dsize=(int(objectB.shape[0] * ratioResize), int(objectB.shape[1] * ratioResize))).astype("bool")


    # 1) Get the (x,y) coordinates of the points in each of the 2 objects
    pointsObjectA = np.argwhere(objectA == True)
    pointsObjectB = np.argwhere(objectB == True)

    # 2) Compute the angles' histogram between each couple of points (pA, pB)
    angleHist = dict()

    for pA in pointsObjectA:
        for pB in pointsObjectB:
            dy = pB[1] - pA[1]
            dx = pB[0] - pA[0]
            angleInRadians = math.atan2(dy, dx) # in interval [-pi ; +pi]
            angleHist[angleInRadians] = angleHist.get(angleInRadians, 0) + 1

    # 3) Normalize the angles' histogram (so the values are < 1) (they correspond to frequencies of the angles)
    nbCouplesPoints = len(pointsObjectA) * len(pointsObjectB)
    angleHist = {k: v / nbCouplesPoints for k,v in angleHist.items()}

    # 4) Compute the compability of the distribution (the histogram) to each of the 4 fuzzy sets (one for each direction)
    #       Right :     angle in [-pi/2 ; +pi/2]
    #       Left :      angle in [-pi ; -pi/2] u [+pi/2 ; +pi]
    #       Above :     angle in [-pi ; 0]
    #       Under :     angle in [0 ; +pi]
    #   (see the paper "Spatial Organization in 2D images", Fig. 5 in chapter III. A)
    directionsCompatibility = dict()
    for direction in ["Right", "Left", "Above", "Under"]:
        directionsCompatibility[direction] = 0

    for angle, angleFrequency in angleHist.items():
        cos2value = math.cos(angle)**2
        sin2value = math.sin(angle)**2

        # Right or Left
        if angle > -math.pi / 2 and angle < math.pi / 2:
            directionsCompatibility["Right"] += cos2value * angleFrequency
        else:
            directionsCompatibility["Left"] += cos2value * angleFrequency

        # Above or Under
        if angle < 0:
            directionsCompatibility["Above"] += sin2value * angleFrequency
        else:
            directionsCompatibility["Under"] += sin2value * angleFrequency

    #print(directionsCompatibility)

    return list(directionsCompatibility.values())

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
    angles = angle_descriptor(objects)
    return rlm1, rlm2, force, dist1, dist2, angles


def image_processing_v3(imagename, background, step, force_type, 
                        computeRLM: bool = False, computeForce: bool = False,
                        computeDist: bool = False, computeAngles: bool = False):
    rlm1, rlm2, force = [], [], []
    dist1, dist2, angles = [], [], []

    objects = ImageRLM.image_segmentation(imagename, background)
    x, y = ImageRLM.center_point(objects)
    lines, diameters = ImageRLM.lines_diameters(objects, x, y, step * math.pi / 180)

    if computeRLM:      rlm1, rlm2 = ImageRLM.radial_line_model(lines, objects)
    if computeForce:    force = ImageRLM.forces(objects, diameters, force_type)
    if computeDist:     dist1, dist2 = distance_descriptors(objects, x, y, lines)
    if computeAngles:   angles = angle_descriptor(objects)
    
    return rlm1, rlm2, force, dist1, dist2, angles