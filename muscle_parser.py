from lxml import etree
from dm_control.utils import xml_tools
from stl import mesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import os
import sys
import csv

from tendon_routing import configs, reserve


def array_to_string(array):
  return ' '.join(['%8g' % num for num in array])


def calculate_transformation(element):

  def get_rotation(element):
    if 'quat' in element.keys():
      r = R.from_quat(element.get('quat'))
    elif 'euler' in element.keys():
      r = R.from_euler('xyz', element.get('euler'))
    elif 'axisangle' in element.keys():
      raise NotImplementedError
    elif 'xyaxes' in element.keys():
      raise NotImplementedError
    elif 'zaxis' in element.keys():
      raise NotImplementedError
    else:
      r = R.identity()
    return r.as_matrix()

  # Calculate all transformation matrices from root until this element
  all_transformations = []
  while element.keys():
    if "pos" in element.keys():
      pos = np.array(element.get('pos').split(), dtype=np.float)
      rot = get_rotation(element)
      all_transformations.append(
        np.vstack((np.hstack((rot, pos.reshape([-1, 1]))), np.array([0, 0, 0, 1])))
      )
    element = element.getparent()

  # Apply all transformations
  T = np.eye(4)
  for transformation in reversed(all_transformations):
    T = np.matmul(T, transformation)

  # pos = np.array(element.get('pos').split(), dtype=np.float)
  # rot = get_rotation(element)
  # T = np.vstack((np.hstack((rot, pos.reshape([-1, 1]))), np.array([0, 0, 0, 1])))

  return T


def parse_tendon(config, mjcf):

  # Create the tendon while parsing sites
  spatial = etree.Element('spatial', name=f'{config.name}_tendon')

  # Bookkeeping for branching
  branch_start = []
  divisor = [1]

  def parse_tendon_route(targets):

    for target in targets:

      if isinstance(target, list):

        # The tendon is branching
        divisor.append(divisor[-1] * len(target))
        branch_start.append(spatial.getchildren()[-1])

        # Go through each branch separately (depth first)
        for branch in target:
          # Add the pulley
          spatial.append(etree.Element('pulley', divisor=f'{divisor[-1]}'))
          spatial.append(deepcopy(branch_start[-1]))
          parse_tendon_route(branch)

        branch_start.pop()
        divisor.pop()

      else:

        # Load stl file of site
        m = mesh.Mesh.from_file(target.mesh_file)

        # Use midpoint of mesh for site position
        T_midpoint = np.eye(4)
        T_midpoint[:3, 3] = m.vectors.mean(axis=(0, 1))

        # Estimate site's position relative to body's position
        body = xml_tools.find_element(mjcf, 'geom', target.geom).getparent()
        T_body = calculate_transformation(body)
        T_site = np.matmul(np.linalg.inv(T_body), T_midpoint)

        # Create the site
        site = etree.Element('site', name=target.name, pos=array_to_string(T_site[:3, 3]))
        site.attrib["class"] = 'connector'

        # Add the site into body; either after existing sites or append to end
        if body.findall('site'):
          # Don't add the site if it already exists
          site_exists = False
          for s in body.findall('site'):
            if s.get('name') == target.name:
              site_exists = True
              break
          if not site_exists:
            s.addnext(site)

        else:
          body.append(site)

        # Add site to tendon
        spatial.append(etree.Element('site', site=target.name))

  # Parse the tendon
  parse_tendon_route(config.sites)

  # Return tendon definition
  return spatial


def main(scale_file="acsa.csv", global_scale=1.0):

  # Make sure global_scale is a number, and scale_file exists
  global_scale = float(global_scale)
  scale_file = os.path.join(os.path.dirname(__file__), 'muscle_scales', scale_file)
  assert os.path.isfile(scale_file), "Given scale file doesn't exist"

  # Get suite directory
  suite_dir = os.path.join(os.path.dirname(__file__), '../dm_control/dm_control/suite')

  # Read the original dog model xml file and parse it
  xml_file = os.path.join(suite_dir, 'dog.xml')
  with open(xml_file) as f:
    xml_string = f.read()
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)

  # Remove existing tendons
  tendon = mjcf.find('tendon')
  for element in tendon.getchildren():
    element.getparent().remove(element)

  # Remove existing actuators
  actuator = mjcf.find('actuator')
  for element in actuator.getchildren():
    element.getparent().remove(element)

  # Read muscle scale config file
  scales = dict()
  with open(scale_file) as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
      scales[row[0]] = float(row[1])

  # Go through muscle configs
  for mtu in configs:

    # Get the tendon for this muscle
    spatial = parse_tendon(mtu, mjcf)

    # Add to collection of tendons
    tendon.append(spatial)

    # Create muscle element
    muscle = etree.Element('muscle', name=mtu.name, tendon=spatial.get('name'))
    muscle.attrib["class"] = "muscle"

    # Add muscle scale
    if mtu.name not in scales:
      print(f"Muscle {mtu.name} is missing scale")
    else:
      muscle.attrib['scale'] = str(global_scale*scales[mtu.name])

    # Add muscle to actuators
    actuator.append(muscle)

  # Delete existing top-level general defaults (dynprm and gainprm set here mess up defaults for muscle actuators even
  # when dynprm and gainprm are set again in another another lower level class), and set up new defaults for muscle
  # class
  general = mjcf.find("default").find("general")
  if general is not None:
    general.getparent().remove(general)
  muscle = mjcf.find("default").find("default[@class='muscle']")
  if muscle is not None:
    muscle.append(etree.Element("muscle", ctrllimited="true", ctrlrange="0 1", group="0"))
  else:
    muscle = etree.Element("default")
    muscle.attrib["class"] = "muscle"
    muscle.append(etree.Element("muscle", ctrllimited="true", ctrlrange="0 1", group="0"))
    muscle.append(etree.Element("geom", contype="0", conaffinity="0", group="4", rgba="0.5 0 0 1"))
    mjcf.find("default").append(muscle)

  # For testing, set gravity to zero and viscosity to one
  if False:
    option = mjcf.find("option")
    option.attrib["gravity"] = "0 0 0"
    option.attrib["viscosity"] = "1"

  # Save the model into a new file
  new_file = os.path.join(suite_dir, 'dog_muscles.xml')
  mjcf.getroottree().write(new_file, encoding='utf-8', xml_declaration=False, pretty_print=True)

if __name__ == "__main__":
    main(*sys.argv[1:])
