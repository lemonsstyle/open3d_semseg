"""
使用 Semantic3D 预训练的 RandLA-Net 模型对自定义 .ply 点云进行语义分割
"""

import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    original_points = points.copy()  # 保存原始点云用于最后的可视化

    # 2. 准备输入数据格式
    data = {
        'point': points,
        'feat': colors,
        'label': np.zeros((points.shape[0],), dtype=np.int32)  # 虚拟标签
    }

    # 3. 加载配置文件
    print(f"\n正在加载配置文件: {config_path}")
    cfg = _ml3d.utils.Config.load_from_file(config_path)

    # 4. 创建模型
    print("正在创建 RandLA-Net 模型...")
    model = ml3d.models.RandLANet(**cfg.model)

    # 5. 加载预训练权重
    print(f"正在加载预训练权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.device = device  # 设置模型的 device 属性
    model.to(device)
    model.eval()
    print("模型加载完成")

    # 6. 数据预处理
    print("\n正在预处理数据...")
    processed_data = model.preprocess(data, {'split': 'test'})

    # 7. 创建推理数据集
    from ml3d.datasets import InferenceDummySplit
    from ml3d.torch.dataloaders import TorchDataloader, get_sampler, DefaultBatcher

    infer_dataset = InferenceDummySplit(data)
    infer_sampler = infer_dataset.sampler

    def get_cache(attr):
        return processed_data

    infer_split = TorchDataloader(
        dataset=infer_dataset,
        preprocess=model.preprocess,
        transform=model.transform,
        sampler=infer_sampler,
        use_cache=False,
        cache_convert=get_cache
    )

    batcher = DefaultBatcher()
    infer_loader = DataLoader(
        infer_split,
        batch_size=cfg.pipeline.get('test_batch_size', 1),
        sampler=get_sampler(infer_sampler),
        collate_fn=batcher.collate_fn
    )

    model.trans_point_sampler = infer_sampler.get_point_sampler()

    # 8. 运行推理
    print("\n开始推理...")
    curr_cloud_id = -1
    test_probs = []
    ori_test_probs = []
    ori_test_labels = []
    pbar = None
    pbar_update = 0

    with torch.no_grad():
        for step, inputs in enumerate(infer_loader):
            # 初始化进度条
            if curr_cloud_id != infer_sampler.cloud_id:
                curr_cloud_id = infer_sampler.cloud_id
                num_points = infer_sampler.possibilities[curr_cloud_id].shape[0]
                pbar = tqdm(total=num_points, desc=f"test {curr_cloud_id}/{len(infer_sampler.dataset)}")
                pbar_update = 0
                test_probs.append(
                    np.zeros(shape=[num_points, model.cfg.num_classes], dtype=np.float16)
                )

            # 前向传播
            if hasattr(inputs['data'], 'to'):
                inputs['data'].to(device)
            results = model(inputs['data'])

            # 更新概率
            test_probs[curr_cloud_id] = model.update_probs(
                inputs, results, test_probs[curr_cloud_id]
            )

            # 更新进度条
            this_possibility = infer_sampler.possibilities[curr_cloud_id]
            end_threshold = 0.5
            pbar.update(
                this_possibility[this_possibility > end_threshold].shape[0] - pbar_update
            )
            pbar_update = this_possibility[this_possibility > end_threshold].shape[0]

            # 检查是否完成
            if this_possibility[this_possibility > end_threshold].shape[0] == this_possibility.shape[0]:
                proj_inds = processed_data.get('proj_inds', None)
                if proj_inds is None:
                    proj_inds = np.arange(test_probs[curr_cloud_id].shape[0])

                test_labels = np.argmax(test_probs[curr_cloud_id][proj_inds], 1)
                ori_test_probs.append(test_probs[curr_cloud_id][proj_inds])
                ori_test_labels.append(test_labels)
                pbar.close()
                break

    # 9. 获取预测结果
    predict_labels = ori_test_labels[0]
    predict_scores = ori_test_probs[0]

    print(f"\n推理完成!")
    print(f"预测标签形状: {predict_labels.shape}")
    print(f"预测分数形状: {predict_scores.shape}")
    print(f"唯一的类别标签: {np.unique(predict_labels)}")
    print(f"\n各类别点数统计:")
    unique, counts = np.unique(predict_labels, return_counts=True)

    # Semantic3D 类别名称
    class_names = {
        0: "unlabeled",
        1: "man-made terrain",
        2: "natural terrain",
        3: "high vegetation",
        4: "low vegetation",
        5: "buildings",
        6: "hard scape",
        7: "scanning artefacts",
        8: "cars"
    }

    for label, count in zip(unique, counts):
        class_name = class_names.get(label, "unknown")
        print(f"  类别 {label} ({class_name}): {count} 个点 ({count/len(predict_labels)*100:.2f}%)")

    # 10. 可视化和保存结果
    visualize_and_save_results(original_points, predict_labels, predict_scores, output_path)

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
        score_file = output_ply.replace('.ply', '_scores.npy')
        np.save(score_file, scores)
        print(f"已保存置信度文件到: {score_file}")

    # 可视化
    print("\n正在打开可视化窗口...")
    print("\n类别颜色说明:")
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
