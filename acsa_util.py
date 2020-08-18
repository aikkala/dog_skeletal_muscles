import scipy
import scipy.stats
import numpy as np
import time
import os
from tendon_routing import configs, reserve
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as pp
import matplotlib
matplotlib.use('TkAgg')

"""This script estimates plane locations for muscles using PCA, which then need to be manually adjusted, after which
 they can be used used to calculate the anatomical cross-sectional area for muscles"""


def generate_coordinate_system(normal):
  """Generate an arbitrary coordinate frame perpendicular to given normal

  Args:
    normal (3x1 array): Normal of a plane

  Returns:
    3x3 array: Basis vectors defining a coordinate frame"""

  # Make sure given normal is normalised
  z = normal / np.sqrt(np.dot(normal, normal))

  # Generate first arbitrary vector that is perpendicular: choose first two dimensions as one
  x = np.array([1, 1, -(normal[0] + normal[1])/normal[2]])
  x /= np.sqrt(np.dot(x, x))

  # Generate second vector that is perpendicular to z and x
  y = np.cross(z, x)
  y /= np.sqrt(np.dot(y, y))

  return np.vstack((x, y, z)).T

def distance_to_plane(plane_eq, points):
  """Estimate 'points' distance from plane

  Args:
    points (3xN array): xyz-coordinates of N points

  Returns:
    Nx1 vector: distances"""

  return np.matmul(plane_eq[:3], points) + plane_eq[3]


def estimate_planes(visualise):
  """Estimates plane location using PCA

  Args:
    visualise (bool): Visualise plane and muscle if set to True

  Returns:
    none"""

  # Ask whether user wants to estimate planes again (and save in a new folder), or go through previously processed
  # muscles (in case new ones have been added)
  inp = input(
    "Input folder name. If this folder exists only muscles that haven't been previously processed\n"
    "will be processed. If left empty a new folder will be created.\n")

  if inp == "":
    # Create a new output folder so we don't accidentally overwrite previous planes (in case they have been manually
    # adjusted)
    folder_name = f"{int(time.time())}"
  else:
    # Use given folder name, make sure it exists
    folder_name = inp

  # Make sure folder exists
  output_folder = os.path.join(os.path.dirname(__file__), "acsa_planes", folder_name)
  os.makedirs(output_folder, exist_ok=True)

  if visualise:
    # Initialise figure
    figure = pp.figure()
    ax = figure.add_subplot(111, projection='3d')
    pp.ion()
    pp.show()

  # Loop over each mesh file for muscles, and calculate a rough estimation for the plane where cross-sectional
  # area could be estimated from
  for mtu in configs:
    for muscle_mesh_file, output_file in zip(mtu.muscle_mesh_file, mtu.plane_mesh_file):

      # Some muscles are used twice (e.g. when there's only one muscle), no need to process them again; also, we want
      # to skip muscles that have already been processed previously
      if os.path.isfile(output_file):
        continue

      # Get the plane mesh and set its midpoint to origin
      plane_mesh = mesh.Mesh.from_file('plane.stl')
      plane_mesh.vectors -= np.mean(plane_mesh.vectors, axis=0)

      # Get muscle mesh
      muscle_mesh = mesh.Mesh.from_file(muscle_mesh_file)

      # Use midpoints muscle_mesh triangles for PCA
      data = muscle_mesh.vectors.mean(axis=1)
      mean = np.mean(data, axis=0)

      # Center data and get covariance matrix
      cov = np.cov(data-mean, rowvar=False)

      # Get eigenvector corresponding to direction of greatest variance, which is the normal of the plane we want for
      # calculating the cross-sectional area
      _, normal = scipy.linalg.eigh(cov, subset_by_index=[2, 2])

      # Align plane_mesh's normal and the normal we got from PCA
      v = np.cross(plane_mesh.normals[0, :], normal.squeeze())
      c = np.dot(plane_mesh.normals[0, :], normal)
      if c != -1:
        # Get the rotation between the normals
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + np.dot(vx, vx)*(1/(1+c))
      else:
        # Vectors are parallel, which is all we need
        R = np.eye(3)

      # Translate and rotate plane_mesh
      T = np.vstack((np.hstack((R, mean.reshape([-1, 1]))), np.array([0, 0, 0, 1])))
      for idx, vector in enumerate(plane_mesh.vectors):
        vector = np.hstack((vector, np.array([[1, 1, 1]]).T))
        plane_mesh.vectors[idx, :, :] = np.matmul(T, vector.T)[:3, :3].T

      # Save the plane_mesh for further (manual) revision
      plane_mesh.save(output_file)

      if visualise:
        # Plot muscle and plane
        muscle_poly = mplot3d.art3d.Poly3DCollection(muscle_mesh.vectors)
        muscle_poly.set_edgecolor('k')
        muscle_poly.set_facecolor('b')
        muscle_poly.set_alpha(0.1)
        plane_poly = mplot3d.art3d.Poly3DCollection(plane_mesh.vectors)
        plane_poly.set_facecolor('r')
        plane_poly.set_alpha(0.1)
        plane_poly.set_edgecolor('k')
        ax.add_collection3d(muscle_poly)
        ax.add_collection3d(plane_poly)

        # Scale axes
        all_points = np.concatenate((np.concatenate(muscle_mesh.vectors), np.concatenate(plane_mesh.vectors)))
        ax.set_xlim3d(min(all_points[:, 0]), max(all_points[:, 0]))
        ax.set_ylim3d(min(all_points[:, 1]), max(all_points[:, 1]))
        ax.set_zlim3d(min(all_points[:, 2]), max(all_points[:, 2]))

        # Wait for keypress before continuing
        keyboard_click = False
        while not keyboard_click:
          keyboard_click = pp.waitforbuttonpress()
        ax.collections.clear()


def calculate_polygon_area(corners):
  """Calculate 2D polygon area with given corners. Corners are assumed to be ordered.

  Args:
    corners (Nx2 array): xy-coordinates of N corner points

  Returns:
    float: polygon area"""

  # Use the shoelace algorithm to calculate polygon's area
  psum = 0
  nsum = 0

  npoints = corners.shape[0]

  for i in range(npoints):
      j = (i + 1) % npoints
      psum += corners[i, 0] * corners[j, 1]
      nsum += corners[j, 0] * corners[i, 1]

  return abs(1/2*(psum - nsum))

def order_triangles(triangles):
  """Finds the cycle formed by given triangles. We assume all triangles share two points with neighbouring triangles.

  Start from randomly chosen (first) triangle, and find the neighbouring triangles until we complete a cycle. We should
  be able to start from any random triangle since a plane always splits two sides of a triangle (unless the plane goes
  exactly through one of the corners)

  Args:
    triangles (Nx3x3 array): N triangles defined by three points given in xyz coordinates

  Returns:
    Nx1 vector: Order of triangles"""

  # Helper function to determine whether two triangles are neighbours
  def are_neighbours(i, j):
    diff1 = np.all(np.isclose(triangles[i, 0, :], triangles[j, :, :]), axis=1)
    diff2 = np.all(np.isclose(triangles[i, 1, :], triangles[j, :, :]), axis=1)
    diff3 = np.all(np.isclose(triangles[i, 2, :], triangles[j, :, :]), axis=1)

    # If given triangles share two points, they are neighbours
    return sum(diff1 | diff2 | diff3) == 2

  # Start from first triangle
  visited = [0]

  def recursive_search(current, remaining):

    if len(visited) > 2 and are_neighbours(current, visited[0]):
      return True

    # Find next neighbour
    for i in remaining:

      if are_neighbours(i, visited[-1]):

        # Follow this branch until end
        visited.append(i)
        if recursive_search(i, set(range(triangles.shape[0])) - set(visited)):
          return True
        else:
          # Branch ended and loop wasn't completed, move on to next branch
          visited.pop()

    return False

  assert recursive_search(0, set(range(triangles.shape[0])) - {0}), "Couldn't find a cycle"

  # Return ordered cycle
  return visited

def estimate_acsa(visualise):
  """Estimate anatomical cross-sectional areas for all muscles in tendon_routing.configs and save them into a config
  file

  Args:
    visualise (bool): Use visualisations

  Returns:
    none"""

  # Use this arbitrary value to scale cross-sectional areas
  scale = 1e7

  # Make sure output folder exists
  output_folder = os.path.join(os.path.dirname(__file__), "muscle_scales")
  os.makedirs(output_folder, exist_ok=True)

  # Ask user for filename
  output_file = input("Input filename for muscle scales\n")

  if visualise:
    figure1 = pp.figure()
    figure2 = pp.figure()
    ax1 = figure1.add_subplot(111, projection='3d')
    ax2 = figure2.add_subplot(111)
    pp.ion()
    pp.show()

  # Loop through all muscles / mesh files
  muscle_names = []
  acsas = []
  for mtu in configs:

    acsa = 0

    for muscle_mesh_file, plane_mesh_file in zip(mtu.muscle_mesh_file, mtu.plane_mesh_file):

      # Open both files
      muscle_mesh = mesh.Mesh.from_file(muscle_mesh_file)
      plane_mesh = mesh.Mesh.from_file(plane_mesh_file)

      if visualise:
        # Plot muscle and plane polygons
        muscle_poly = mplot3d.art3d.Poly3DCollection(muscle_mesh.vectors)
        muscle_poly.set_edgecolor('k')
        muscle_poly.set_facecolor('b')
        muscle_poly.set_alpha(0.1)
        plane_poly = mplot3d.art3d.Poly3DCollection(plane_mesh.vectors)
        plane_poly.set_facecolor('r')
        plane_poly.set_alpha(0.1)
        plane_poly.set_edgecolor('k')
        ax1.add_collection3d(muscle_poly)
        ax1.add_collection3d(plane_poly)

        # Scale axes
        all_points = np.concatenate((np.concatenate(muscle_mesh.vectors), np.concatenate(plane_mesh.vectors)))
        ax1.set_xlim3d(min(all_points[:, 0]), max(all_points[:, 0]))
        ax1.set_ylim3d(min(all_points[:, 1]), max(all_points[:, 1]))
        ax1.set_zlim3d(min(all_points[:, 2]), max(all_points[:, 2]))

      # Form the plane equation; all normals should point into the same direction but let's take mean anyway
      plane_normal = plane_mesh.normals.mean(axis=0)
      plane_mean = plane_mesh.vectors.mean(axis=(0, 1))
      plane_eq = np.array([plane_normal[0], plane_normal[1], plane_normal[2], -np.dot(plane_normal, plane_mean)])

      # Find an arbitrary coordinate system on the plane (such that two of the basis vectors lie on the plane, and the third
      # one is in direction of the normal of the plane) for later use
      basis = generate_coordinate_system(plane_normal)

      # Calculate which muscle_mesh triangles intersect the plane, project them onto the plane;
      # these projected points form corners of a polygon that represents the cross-sectional area of the muscle
      corners = []
      triangles = []
      for triangle in muscle_mesh.vectors:
        d = distance_to_plane(plane_eq, triangle.T)

        # If any of the distances is zero, or sign of one of them is different from the others,
        # then the plane intersects this triangle
        if not np.any(np.all(np.sign(d)==-1) or np.all(np.sign(d)==1)):
          # Project the points onto the plane
          projected = (triangle - np.outer(d, plane_normal)).T

          if visualise:
            # Plot points of intersecting triangles and their projections on the plane
            ax1.scatter(triangle[:, 0], triangle[:, 1], triangle[:, 2], c='g', s=100)
            ax1.scatter(projected[0, :], projected[1, :], projected[2, :], c='r', s=100)

          # We'll further convert to an arbitrary coordinate frame on the plane
          # to get 2D points
          corners.append(np.matmul(basis[:, :2].T, projected))

          # We'll need the original 3D coordinates as well to order the corners
          triangles.append(triangle)

      # And then estimate the cross-sectional area (order them first and use only midpoints of triangles)
      order = order_triangles(np.asarray(triangles))
      corners = np.asarray(corners)[order, :, :].mean(axis=2)

      # Calculate the area
      acsa += calculate_polygon_area(corners)

      if visualise:

        # Plot 2D projected corner points of the polygon, color-coded such that first point is red and last point
        # is blue
        ax2.scatter(corners[:, 0], corners[:, 1], c=pp.get_cmap('coolwarm')(np.linspace(0, 1, corners.shape[0])))
        xmin, xmax = min(corners[:, 0]), max(corners[:, 0])
        ymin, ymax = min(corners[:, 1]), max(corners[:, 1])
        ax2.set_xlim(xmin - 0.2*abs(xmax-xmin), xmax + 0.2*abs(xmax-xmin))
        ax2.set_ylim(ymin - 0.2*abs(ymax-ymin), ymax + 0.2*abs(ymax-ymin))

        # Wait for keypress before continuing
        keyboard_click = False
        while not keyboard_click:
          keyboard_click = pp.waitforbuttonpress()
        ax1.collections.clear()
        ax2.collections.clear()

    # Save scale for this muscle
    muscle_names.append(mtu.name)
    acsas.append(scale*acsa*mtu.scale_factor)

  # Write muscle scale config file
  with open(os.path.join(output_folder, output_file), 'w') as f:
    for muscle_name, acsa in zip(muscle_names, acsas):
      f.write(f"{muscle_name} {acsa}\n")

def main():

  visualise = False
  inp = ""
  while inp not in {"1", "2", "q"}:
    print("Visualisations: " + ("on" if visualise else "off"))
    inp = input("Choose\n"
                "  1) Estimate planes for calculating anatomical cross-sectional areas\n"
                "  2) Calculate anatomical cross-sectional areas\n"
                "  t) Toggle visualisations on/off\n"
                "  q) Quit\n")

    if inp == "t":
      visualise = not visualise

  if inp == "1":
    estimate_planes(visualise=visualise)
  elif inp == "2":
    estimate_acsa(visualise=visualise)

main()