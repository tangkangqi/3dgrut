#!/usr/bin/env python3
"""
3DGS PLY加载和多实例3DGRT渲染脚本
支持从PLY文件加载模型并创建10个实例化的3DGRT渲染

使用方法:
    python multi_instance_3dgrt_render.py --ply_path model.ply --num_instances 10
"""

import argparse
import numpy as np
import torch
import torchvision
from pathlib import Path
from omegaconf import OmegaConf
from typing import List, Tuple
import time

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.datasets.protocols import Batch


class MultiInstanceRenderer:
    """3DGS多实例渲染器"""
    
    def __init__(self, ply_path: str, config_path: str, num_instances: int = 10):
        """
        初始化渲染器
        
        Args:
            ply_path: PLY文件路径
            config_path: 配置文件路径
            num_instances: 要创建的实例数
        """
        self.ply_path = ply_path
        self.config_path = config_path
        self.num_instances = num_instances
        self.models = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 初始化MultiInstanceRenderer")
        print(f"   - PLY文件: {ply_path}")
        print(f"   - 配置: {config_path}")
        print(f"   - 实例数: {num_instances}")
        print(f"   - 设备: {self.device}")
    
    def load_config(self) -> dict:
        """加载配置文件"""
        print("\n📋 加载配置...")
        conf = OmegaConf.load(self.config_path)
        print(f"✅ 配置已加载")
        return conf
    
    def create_model_instances(self) -> List[MixtureOfGaussians]:
        """
        创建多个独立的模型实例
        
        Returns:
            模型实例列表
        """
        print(f"\n🚀 创建 {self.num_instances} 个模型实例...")
        
        conf = self.load_config()
        models = []
        
        # 创建第一个实例
        master_model = MixtureOfGaussians(conf, scene_extent=1.0)
        master_model.init_from_ply(self.ply_path, init_model=True)
        master_model.build_acc(rebuild=True)
        models.append(master_model)
        print(f"✅ 实例 1/{self.num_instances} - {master_model.num_gaussians:,} 个高斯")
        
        # 克隆其他实例（更高效）
        for i in range(1, self.num_instances):
            cloned_model = master_model.clone()
            cloned_model.build_acc(rebuild=True)
            models.append(cloned_model)
            
            # 每2个实例打印一次进度
            if (i + 1) % 2 == 0 or i == self.num_instances - 1:
                print(f"✅ 实例 {i+1}/{self.num_instances} - {cloned_model.num_gaussians:,} 个高斯")
        
        self.models = models
        return models
    
    def apply_grid_layout(self):
        """
        在3x3+1网格中放置实例
        
        布局:
        (0,0) (1,0) (2,0)
        (0,1) (1,1) (2,1)
        (0,2) (1,2) (2,2)
        (1,3)
        """
        print("\n🎯 应用网格布局...")
        
        spacing = 2.0
        offset = -(spacing * 1.5)
        
        positions = [
            (0*spacing + offset, 0*spacing + offset, 0),
            (1*spacing + offset, 0*spacing + offset, 0),
            (2*spacing + offset, 0*spacing + offset, 0),
            (0*spacing + offset, 1*spacing + offset, 0),
            (1*spacing + offset, 1*spacing + offset, 0),  # 中心
            (2*spacing + offset, 1*spacing + offset, 0),
            (0*spacing + offset, 2*spacing + offset, 0),
            (1*spacing + offset, 2*spacing + offset, 0),
            (2*spacing + offset, 2*spacing + offset, 0),
            (1*spacing + offset, 3*spacing + offset, 0),
        ]
        
        for i, (model, pos) in enumerate(zip(self.models, positions)):
            self._apply_transformation(model, pos, scale=0.5)
        
        print(f"✅ 网格布局应用完成")
    
    def apply_circular_layout(self):
        """在圆形布局中放置实例"""
        print("\n🎯 应用圆形布局...")
        
        radius = 3.0
        
        for i, model in enumerate(self.models):
            angle = 2 * np.pi * i / self.num_instances
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0.0
            
            self._apply_transformation(model, (x, y, z), scale=0.3)
        
        print(f"✅ 圆形布局应用完成")
    
    def apply_random_layout(self, seed: int = 42):
        """随机放置实例"""
        print("\n🎯 应用随机布局...")
        
        np.random.seed(seed)
        bbox_range = 5.0
        
        for i, model in enumerate(self.models):
            pos = (
                np.random.uniform(-bbox_range, bbox_range),
                np.random.uniform(-bbox_range, bbox_range),
                np.random.uniform(-0.5, 0.5)
            )
            scale = np.random.uniform(0.2, 0.8)
            
            self._apply_transformation(model, pos, scale=scale)
        
        print(f"✅ 随机布局应用完成")
    
    @staticmethod
    def _apply_transformation(
        model: MixtureOfGaussians,
        position: Tuple[float, float, float],
        scale: float = 1.0
    ):
        """
        对模型应用刚体变换
        
        Args:
            model: 模型实例
            position: 平移向量
            scale: 缩放因子
        """
        with torch.no_grad():
            # 平移
            pos_tensor = torch.tensor(position, dtype=torch.float32, device=model.device)
            model.positions.data = model.positions.data + pos_tensor
            
            # 缩放
            if scale != 1.0:
                model.positions.data = model.positions.data * scale
                scale_factor = torch.log(torch.tensor(scale, dtype=torch.float32, device=model.device))
                model.scale.data = model.scale.data + scale_factor
    
    def create_camera_pose(
        self,
        position: Tuple[float, float, float] = (0, 1, 3),
        look_at: Tuple[float, float, float] = (0, 0, 0),
        up: Tuple[float, float, float] = (0, 1, 0)
    ) -> torch.Tensor:
        """
        创建相机姿态矩阵（世界坐标系）
        
        Args:
            position: 相机位置
            look_at: 看向的目标点
            up: 上方向向量
        
        Returns:
            [4, 4] 相机到世界变换矩阵
        """
        pos = np.array(position, dtype=np.float32)
        target = np.array(look_at, dtype=np.float32)
        up_vec = np.array(up, dtype=np.float32)
        
        # 计算相机坐标系的三个轴
        forward = target - pos
        forward = forward / (np.linalg.norm(forward) + 1e-6)
        
        right = np.cross(forward, up_vec)
        right = right / (np.linalg.norm(right) + 1e-6)
        
        up_vec = np.cross(right, forward)
        up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-6)
        
        # 构造4x4矩阵
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up_vec
        c2w[:3, 2] = -forward
        c2w[:3, 3] = pos
        
        return torch.tensor(c2w, dtype=torch.float32, device=self.device)
    
    def render_single_instance(
        self,
        instance_idx: int,
        camera_pose: torch.Tensor,
        intrinsics: Tuple[float, float, float, float] = (1000, 1000, 256, 256),
        output_path: Path = None
    ) -> torch.Tensor:
        """
        渲染单个实例
        
        Args:
            instance_idx: 实例索引
            camera_pose: 相机姿态
            intrinsics: 相机内参 (fx, fy, cx, cy)
            output_path: 输出文件路径
        
        Returns:
            渲染的RGB图像
        """
        model = self.models[instance_idx]
        
        # 这里需要根据项目的实际Batch协议创建批次
        # 以下是简化版本，实际使用需要调整
        
        with torch.no_grad():
            try:
                # 尝试创建批次数据
                # 注意：这需要根据你的项目实际情况调整
                batch = self._create_dummy_batch(camera_pose, intrinsics)
                
                # 渲染
                output = model.forward(batch, train=False)
                pred_rgb = output['pred_rgb'].squeeze(0)  # [H, W, 3]
                
                # 保存图像
                if output_path:
                    torchvision.utils.save_image(
                        pred_rgb.permute(2, 0, 1).clamp(0, 1),
                        str(output_path)
                    )
                
                return pred_rgb
            
            except Exception as e:
                print(f"⚠️  实例 {instance_idx+1} 渲染失败: {str(e)}")
                return None
    
    def _create_dummy_batch(
        self,
        camera_pose: torch.Tensor,
        intrinsics: Tuple[float, float, float, float] = (1000, 1000, 256, 256)
    ) -> Batch:
        """
        创建一个虚拟批次数据
        
        注意：这是一个简化版本，实际使用需要根据项目的Batch协议调整
        """
        fx, fy, cx, cy = intrinsics
        H, W = 512, 512
        
        # 创建简单的批次对象
        batch = Batch()
        batch.c2w = camera_pose.unsqueeze(0)  # [1, 4, 4]
        batch.image_size = (H, W)
        
        return batch
    
    def render_all_instances(
        self,
        output_dir: str = "renders_output",
        layout: str = "grid",
        save_images: bool = True
    ):
        """
        渲染所有实例
        
        Args:
            output_dir: 输出目录
            layout: 布局方式 ('grid', 'circle', 'random')
            save_images: 是否保存图像
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 开始渲染所有实例...")
        
        # 应用布局
        if layout == "grid":
            self.apply_grid_layout()
        elif layout == "circle":
            self.apply_circular_layout()
        elif layout == "random":
            self.apply_random_layout()
        else:
            raise ValueError(f"未知的布局方式: {layout}")
        
        # 创建相机姿态
        camera_pose = self.create_camera_pose(
            position=(0, 1, 3),
            look_at=(0, 0, 0)
        )
        
        # 渲染每个实例
        print(f"\n🎬 渲染进度:")
        start_time = time.time()
        
        for i, model in enumerate(self.models):
            output_file = output_path / f"instance_{i:02d}.png" if save_images else None
            
            try:
                _ = self.render_single_instance(i, camera_pose, output_path=output_file)
                print(f"  ✅ 实例 {i+1:2d}/{self.num_instances} - {model.num_gaussians:,} 个高斯")
            except Exception as e:
                print(f"  ⚠️  实例 {i+1:2d}/{self.num_instances} - 渲染失败: {str(e)}")
        
        elapsed = time.time() - start_time
        
        # 打印统计信息
        print(f"\n📊 渲染统计:")
        print(f"  - 总耗时: {elapsed:.2f} 秒")
        print(f"  - 平均每实例: {elapsed/self.num_instances:.3f} 秒")
        print(f"  - 总高斯数: {sum(m.num_gaussians for m in self.models):,}")
        print(f"  - 平均每实例: {sum(m.num_gaussians for m in self.models)//self.num_instances:,}")
        
        if save_images:
            print(f"  - 输出目录: {output_path}")
            print(f"  - 输出图像数: {self.num_instances}")
        
        print(f"\n✨ 渲染完成！")
    
    def print_model_info(self):
        """打印所有模型的信息"""
        print("\n📊 模型信息:")
        print("-" * 80)
        
        for i, model in enumerate(self.models):
            print(f"\n实例 {i+1}:")
            print(f"  - 高斯数: {model.num_gaussians:,}")
            print(f"  - 位置范围: X[{model.positions[:, 0].min():.2f}, {model.positions[:, 0].max():.2f}]")
            print(f"             Y[{model.positions[:, 1].min():.2f}, {model.positions[:, 1].max():.2f}]")
            print(f"             Z[{model.positions[:, 2].min():.2f}, {model.positions[:, 2].max():.2f}]")
            print(f"  - 密度范围: [{model.density.min():.3f}, {model.density.max():.3f}]")
            print(f"  - 活跃特征: {model.n_active_features}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="3DGS PLY加载和多实例3DGRT渲染",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础用法
  python multi_instance_3dgrt_render.py --ply_path model.ply

  # 自定义参数
  python multi_instance_3dgrt_render.py \\
    --ply_path model.ply \\
    --num_instances 10 \\
    --layout circle \\
    --output_dir ./renders

  # 显示模型信息
  python multi_instance_3dgrt_render.py \\
    --ply_path model.ply \\
    --info_only
        """
    )
    
    parser.add_argument(
        "--ply_path",
        type=str,
        required=True,
        help="3DGS PLY文件路径 (必需)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/apps/colmap_3dgrt.yaml",
        help="配置文件路径 (默认: configs/apps/colmap_3dgrt.yaml)"
    )
    
    parser.add_argument(
        "--num_instances",
        type=int,
        default=10,
        help="要创建的实例数 (默认: 10)"
    )
    
    parser.add_argument(
        "--layout",
        type=str,
        choices=["grid", "circle", "random"],
        default="grid",
        help="实例布局方式 (默认: grid)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="renders_output",
        help="输出目录 (默认: renders_output)"
    )
    
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="不保存渲染图像（仅用于性能测试）"
    )
    
    parser.add_argument(
        "--info_only",
        action="store_true",
        help="仅显示模型信息，不执行渲染"
    )
    
    args = parser.parse_args()
    
    # 创建渲染器
    renderer = MultiInstanceRenderer(
        ply_path=args.ply_path,
        config_path=args.config,
        num_instances=args.num_instances
    )
    
    print("\n" + "=" * 80)
    print("3DGS 多实例3DGRT渲染")
    print("=" * 80)
    
    # 创建模型实例
    renderer.create_model_instances()
    
    # 显示模型信息
    renderer.print_model_info()
    
    # 如果只显示信息则退出
    if args.info_only:
        print("\n✨ 完成（仅显示信息）")
        return
    
    # 渲染所有实例
    try:
        renderer.render_all_instances(
            output_dir=args.output_dir,
            layout=args.layout,
            save_images=not args.no_save
        )
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
