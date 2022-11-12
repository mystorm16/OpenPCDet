"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    # box_points = np.asarray(box3d.get_box_points())
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

# 体素点云可视化 输入：原始点云
def draw_scenes_points2voxel(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    voxel_grid = open3d.geometry.VoxelGrid.create_from_point_cloud(pts, voxel_size=0.6)
    vis.add_geometry(voxel_grid)
    vis.run()
    vis.destroy_window()


# 体素可视化  输入：N,4 coord
def draw_scenes_voxel_a(voxel):
    vis = torch.vstack((voxel[:, 3], voxel[:, 2], voxel[:, 1]))
    draw_scenes(vis.transpose(0, 1), draw_origin=True)
    # draw_spherical_voxels(vis.transpose(0, 1))


# 体素可视化  输入：bs Z Y X  --  9 157 209
def draw_scenes_voxel_b(voxel):
    vis = (voxel == 1).nonzero(as_tuple=False)
    vis = torch.vstack((vis[:, 3], vis[:, 2], vis[:, 1]))
    draw_scenes(vis.transpose(0, 1), draw_origin=True)


# sphere系体素可视化  输入：点云N 3
def draw_spherical_voxels_points(voxel):
    if isinstance(voxel, torch.Tensor):
        voxel = voxel.cpu().numpy()
    visvox = cylinder_uvd2absxyz_np(voxel[:, 0], voxel[:, 1], voxel[:, 2])
    draw_scenes(visvox, draw_origin=True)


# sphere系体素可视化  输入：index N 3
def draw_spherical_voxels_index(voxel):
    if isinstance(voxel, torch.Tensor):
        voxel = voxel.cpu().numpy()
    visvox = cylinder_uvd2absxyz_np(voxel[:, 2], voxel[:, 1], voxel[:, 0])
    draw_scenes(visvox, draw_origin=True)


# sphere系体素可视化  输入：bs 9 157 209
def draw_spherical_voxels_4(voxel):
    vis = (voxel == 1).nonzero(as_tuple=False)
    visvox = cylinder_uvd2absxyz(vis[:, 3], vis[:, 2], vis[:, 1])
    draw_scenes(visvox, draw_origin=True)

    # vis = (voxel == 0).nonzero(as_tuple=False)
    # visvox = cylinder_uvd2absxyz(vis[:, 3], vis[:, 2], vis[:, 1])
    # draw_scenes(visvox, draw_origin=True)