import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import polygonize, unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
from scipy.spatial import Voronoi

def piefun(pa, pb, pc, lc):
    Dab = np.sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
    Dbc = np.sqrt((pc[0]-pb[0])**2+(pc[1]-pb[1])**2)
    Vab = [pa[0]-pb[0], pa[1]-pb[1]]
    Vbc = [pb[0]-pc[0], pb[1]-pc[1]]
    Vab_Vbc = -(Vab[0]*Vbc[0]+Vab[1]*Vbc[1])
    Angle = np.arccos(Vab_Vbc/Dab/Dbc)
    Dtar = lc/2/np.sin(Angle/2)
    Norm_Vab = Vab/Dab
    Norm_Vbc = Vbc/Dbc
    Bisector = Norm_Vab-Norm_Vbc
    Norm_Bisector = Bisector/np.sqrt(Bisector[0]**2+Bisector[1]**2)
    xpie = pb[0]+Norm_Bisector[0]*Dtar
    ypie = pb[1]+Norm_Bisector[1]*Dtar
    return [xpie, ypie]

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

# make up data points
np.random.seed(1234)
points = np.random.rand(100, 2)
points = np.append(points, [[0, 0], [0, 1], [1, 0], [1, 1]], axis=0)
polyset = []

# compute Voronoi tesselation
vor = Voronoi(points)

# plot
regions, vertices = voronoi_finite_polygons_2d(vor)


pts = MultiPoint([Point(i) for i in points])
mask = pts.convex_hull
new_vertices = []
for region in regions:
    polygon = vertices[region]
    shape = list(polygon.shape)
    shape[0] += 1
    p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
    poly = np.array(list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1])))
    new_vertices.append(poly)
    plt.fill(*zip(*poly), alpha=0.4)
    polyset.append(poly)
length = 0.0
GrainData = []
for i in range(len(polyset)):
# for i in range(1):
    # one grain:
    grainset = polyset[i]
    grainset = np.append([polyset[i][-1]], polyset[i], axis=0)
    grainset = np.append(grainset, polyset[i][0:2], axis=0)
    GrainData.append([])
    for j in range(len(polyset[i])+1):
        Pointa = [grainset[j][0], grainset[j][1]]
        Pointb = [grainset[j+1][0], grainset[j+1][1]]
        Pointc = [grainset[j+2][0], grainset[j+2][1]]
        Pointbpie = piefun(Pointa, Pointb, Pointc, length)
        plt.plot(Pointbpie[0], Pointbpie[1], 'ro')
        # print(Pointbpie)
        GrainData[i].append(Pointbpie)

# Generate script files
Matrix = [[0.98,0.98], [0.02,0.02]]
Points = (GrainData)
# 1.0 Part-1 Matrix Drawing
FileName = "204_4.py"
Script = open(FileName, 'w')
Script.write('## ----------------------------------------------------------------')
Script.write('\n')
Script.write('## Abaqus script')
Script.write('\n')
Script.write('## Generated by Python')
Script.write('\n')
Script.write('## ----------------------------------------------------------------')
Script.write('\n')
Script.write("from part import *")
Script.write("\n")
Script.write("from material import *")
Script.write("\n")
Script.write("from section import *")
Script.write("\n")
Script.write("from assembly import *")
Script.write("\n")
Script.write("from step import *")
Script.write("\n")
Script.write("from interaction import *")
Script.write("\n")
Script.write("from load import *")
Script.write("\n")
Script.write("from mesh import *")
Script.write("\n")
Script.write("from optimization import *")
Script.write("\n")
Script.write("from job import *")
Script.write("\n")
Script.write("from sketch import *")
Script.write("\n")
Script.write("from visualization import *")
Script.write("\n")
Script.write("from connectorBehavior import *")
Script.write("\n")
Script.write("mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=1.0)")
Script.write("\n")
Script.write("mdb.models['Model-1'].sketches['__profile__'].rectangle(")
Script.write("\n")
Script.write("".join(["    point1=(", str(Matrix[0][0]), ",", str(Matrix[0][1]), "), point2=(",
               str(Matrix[1][0]), ",", str(Matrix[1][1]), "))"]))
Script.write("\n")
Script.write("mdb.models['Model-1'].Part(dimensionality=TWO_D_PLANAR, name='Part-1', type=")
Script.write("\n")
Script.write("    DEFORMABLE_BODY)")
Script.write("\n")
Script.write("mdb.models['Model-1'].parts['Part-1'].BaseShell(sketch=")
Script.write("\n")
Script.write("    mdb.models['Model-1'].sketches['__profile__'])")
Script.write("\n")
Script.write("del mdb.models['Model-1'].sketches['__profile__']")
Script.write("\n")
# 2.0 Part-1 Grains Drawing
Script.write("mdb.models['Model-1'].ConstrainedSketch(gridSpacing=0.14, name='__profile__',")
Script.write("\n")
Script.write("    sheetSize=5.65, transform=")
Script.write("\n")
Script.write("    mdb.models['Model-1'].parts['Part-1'].MakeSketchTransform(")
Script.write("\n")
Script.write("    sketchPlane=mdb.models['Model-1'].parts['Part-1'].faces[0],")
Script.write("\n")
Script.write("    sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, 0.0)))")
Script.write("\n")
Script.write("mdb.models['Model-1'].parts['Part-1'].projectReferencesOntoSketch(filter=")
Script.write("\n")
Script.write("    COPLANAR_EDGES, sketch=mdb.models['Model-1'].sketches['__profile__'])")
Script.write("\n")
# 2.1 Grain_Point
for i in range(len(Points)):
    for j in range(len(Points[i])-1):
        Script.write("".join(["mdb.models['Model-1'].sketches['__profile__'].Line(point1=(", str(Points[i][j][0]), ", ", str(Points[i][j][1]), "),"]))
        Script.write("\n")
        Script.write("".join(["    point2=(", str(Points[i][j+1][0]), ", ", str(Points[i][j+1][1]), "))"]))
        Script.write("\n")

Script.write("mdb.models['Model-1'].parts['Part-1'].PartitionFaceBySketch(faces=")
Script.write("\n")
Script.write("    mdb.models['Model-1'].parts['Part-1'].faces.getSequenceFromMask(('[#1 ]',")
Script.write("\n")
Script.write("    ), ), sketch=mdb.models['Model-1'].sketches['__profile__'])")
Script.write("\n")
Script.write("del mdb.models['Model-1'].sketches['__profile__']")
Script.write("\n")
