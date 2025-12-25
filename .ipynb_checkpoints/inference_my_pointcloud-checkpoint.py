"""
使用 Semantic3D 预训练的 RandLA-Net 模型对自定义 .ply 点云进行语义分割
"""

import numpy as np
import open3d as o3d
import open3d.ml.torch as ml3d
import argparse
import os


def load_pointcloud(ply_path):
    """加载 .ply 点云文件并提取 xyz 和 rgb 数据"""
    print(f"正在加载点云: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    # 提取 xyz 坐标
    points = np.asarray(pcd.points, dtype=np.float32)
    print(f"点云包含 {points.shape[0]} 个点")

    # 提取 rgb 颜色
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)
        # 如果颜色是 0-1 范围，转换为 0-255
        if colors.max() <= 1.0:
            colors = colors * 255
        print(f"点云包含 RGB 颜色信息")
    else:
        # 如果没有颜色，使用默认值
        colors = np.ones((points.shape[0], 3), dtype=np.float32) * 128
        print(f"警告: 点云没有颜色信息，使用默认灰色")

    return points, colors


def run_inference(ply_path, ckpt_path, config_path, device="cuda", output_path=None):
    """
    对点云进行语义分割推理

    Args:
        ply_path: 输入 .ply 文件路径
        ckpt_path: 预训练权重文件路径
        config_path: 配置文件路径
        device: 使用的设备 ("cuda" 或 "cpu")
        output_path: 输出结果的路径（可选）
    """

    # 1. 加载点云数据
    points, colors = load_pointcloud(ply_path)

    # 2. 准备输入数据格式
    data = {
        'point': points,  # shape: (N, 3) - xyz 坐标
        'feat': colors    # shape: (N, 3) - RGB 特征
    }

    # 3. 加载配置文件
    print(f"\n正在加载配置文件: {config_path}")
    cfg = ml3d.utils.Config.load_from_file(config_path)

    # 4. 创建模型
    print("正在创建 RandLA-Net 模型...")
    model = ml3d.models.RandLANet(**cfg.model)

    # 5. 创建推理 pipeline
    print(f"正在创建推理 pipeline (设备: {device})...")
    pipeline = ml3d.pipelines.SemanticSegmentation(
        model=model,
        device=device,
        **cfg.pipeline
    )

    # 6. 加载预训练权重
    print(f"\n正在加载预训练权重: {ckpt_path}")
    pipeline.load_ckpt(ckpt_path=ckpt_path)

    # 7. 运行推理
    print("\n开始推理...")
    result = pipeline.run_inference(data)

    # 8. 获取预测结果
    predict_labels = result['predict_labels']
    predict_scores = result['predict_scores']

    print(f"\n推理完成!")
    print(f"预测标签形状: {predict_labels.shape}")
    print(f"唯一的类别标签: {np.unique(predict_labels)}")
    print(f"各类别点数统计:")
    unique, counts = np.unique(predict_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count} 个点 ({count/len(predict_labels)*100:.2f}%)")

    # 9. 可视化和保存结果
    visualize_and_save_results(points, predict_labels, predict_scores, output_path)

    return predict_labels, predict_scores


def visualize_and_save_results(points, labels, scores, output_path=None):
    """
    可视化预测结果并保存

    Semantic3D 数据集的类别:
    0: unlabeled / ignored
    1: man-made terrain (人造地形)
    2: natural terrain (自然地形)
    3: high vegetation (高植被)
    4: low vegetation (低植被)
    5: buildings (建筑物)
    6: hard scape (硬景观，如道路)
    7: scanning artefacts (扫描伪影)
    8: cars (汽车)
    """

    # 为每个类别定义颜色 (RGB, 0-1 范围)
    label_to_color = {
        0: [0.5, 0.5, 0.5],      # 灰色 - unlabeled
        1: [0.7, 0.4, 0.4],      # 棕红色 - man-made terrain
        2: [0.6, 0.8, 0.4],      # 黄绿色 - natural terrain
        3: [0.0, 0.6, 0.0],      # 深绿色 - high vegetation
        4: [0.4, 0.8, 0.4],      # 浅绿色 - low vegetation
        5: [0.8, 0.2, 0.2],      # 红色 - buildings
        6: [0.6, 0.6, 0.6],      # 浅灰色 - hard scape
        7: [0.8, 0.0, 0.8],      # 紫色 - scanning artefacts
        8: [0.0, 0.4, 0.8]       # 蓝色 - cars
    }

    # 根据预测标签为点云着色
    colors = np.zeros((points.shape[0], 3))
    for i, label in enumerate(labels):
        if label in label_to_color:
            colors[i] = label_to_color[label]
        else:
            colors[i] = [0, 0, 0]  # 未知类别用黑色

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 保存结果
    if output_path:
        # 保存带颜色的点云
        output_ply = output_path if output_path.endswith('.ply') else output_path + '_segmented.ply'
        o3d.io.write_point_cloud(output_ply, pcd)
        print(f"\n已保存分割结果到: {output_ply}")

        # 保存标签文件
        label_file = output_ply.replace('.ply', '_labels.txt')
        np.savetxt(label_file, labels, fmt='%d')
        print(f"已保存标签文件到: {label_file}")

        # 保存置信度文件
        score_file = output_ply.replace('.ply', '_scores.txt')
        np.savetxt(score_file, scores, fmt='%.6f')
        print(f"已保存置信度文件到: {score_file}")

    # 可视化
    print("\n正在打开可视化窗口...")
    print("类别颜色说明:")
    print("  灰色: unlabeled")
    print("  棕红色: man-made terrain (人造地形)")
    print("  黄绿色: natural terrain (自然地形)")
    print("  深绿色: high vegetation (高植被)")
    print("  浅绿色: low vegetation (低植被)")
    print("  红色: buildings (建筑物)")
    print("  浅灰色: hard scape (硬景观)")
    print("  紫色: scanning artefacts (扫描伪影)")
    print("  蓝色: cars (汽车)")

    o3d.visualization.draw_geometries([pcd],
                                      window_name="语义分割结果",
                                      width=1024,
                                      height=768,
                                      point_show_normal=False)


def main():
    parser = argparse.ArgumentParser(description='使用预训练的 RandLA-Net 对点云进行语义分割')
    parser.add_argument('--input', '-i',
                        required=True,
                        help='输入 .ply 点云文件路径')
    parser.add_argument('--ckpt', '-c',
                        required=True,
                        help='预训练权重文件路径 (.pth)')
    parser.add_argument('--config',
                        default='ml3d/configs/randlanet_semantic3d.yml',
                        help='配置文件路径 (默认: ml3d/configs/randlanet_semantic3d.yml)')
    parser.add_argument('--device',
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help='使用的设备 (默认: cuda)')
    parser.add_argument('--output', '-o',
                        default=None,
                        help='输出文件路径 (可选，不指定则不保存)')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return

    if not os.path.exists(args.ckpt):
        print(f"错误: 权重文件不存在: {args.ckpt}")
        return

    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return

    # 如果没有指定输出路径，使用输入文件名
    if args.output is None:
        args.output = args.input.replace('.ply', '_segmented.ply')

    # 运行推理
    try:
        run_inference(
            ply_path=args.input,
            ckpt_path=args.ckpt,
            config_path=args.config,
            device=args.device,
            output_path=args.output
        )
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
