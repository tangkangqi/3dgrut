# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ctypes
import os
import time
from pathlib import Path

import glfw
import numpy as np
import torch
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from kaolin.render.camera import Camera

from threedgrut_playground.engine import Engine3DGRUT
from threedgrut.utils.logger import logger

# ==================== Shader程序 ====================

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    TexCoord = texCoord;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D texture1;

void main() {
    FragColor = texture(texture1, TexCoord);
}
"""


class InteractiveViewer:
    """
    3DGS交互式查看器（使用3DGRUT引擎）
    实时显示鼠标控制的3D高斯溅射渲染结果
    """
    
    def __init__(self, 
                 gs_object: str,
                 mesh_assets_folder: str = None,
                 default_config: str = "apps/colmap_3dgrt.yaml",
                 envmap_assets_folder: str = None,
                 width: int = 1920, 
                 height: int = 1080,
                 buffer_mode: str = "device2device"):
        """
        Args:
            gs_object: 3DGRUT模型路径 (.pt / .ingp / .ply)
            mesh_assets_folder: Mesh资产文件夹路径
            default_config: 默认配置文件名
            envmap_assets_folder: 环境贴图文件夹路径
            width: 窗口宽度
            height: 窗口高度
            buffer_mode: 缓冲模式 (host2device / device2device)
        """
        self.window_width = width
        self.window_height = height
        self.buffer_mode = buffer_mode
        
        logger.info(f"Loading 3DGRUT model from {gs_object}...")
        
        # 初始化3DGRUT引擎
        if mesh_assets_folder is None:
            mesh_assets_folder = os.path.join(os.path.dirname(__file__), "threedgrut_playground", "assets")
        if envmap_assets_folder is None:
            envmap_assets_folder = os.path.join(os.path.dirname(__file__), "threedgrut_playground", "assets")
            
        self.engine = Engine3DGRUT(
            gs_object=gs_object,
            mesh_assets_folder=mesh_assets_folder,
            default_config=default_config,
            envmap_assets_folder=envmap_assets_folder
        )
        
        logger.info("3DGRUT engine initialized successfully!")
        
        # 配置引擎渲染参数
        self.engine.use_spp = False  # 禁用抗锯齿以加快渲染
        self.engine.use_depth_of_field = False  # 禁用景深
        self.engine.camera_type = "Pinhole"
        self.engine.camera_fov = 45.0
        
        # 计算物体边界盒并自动居中
        self._compute_scene_bounds()
        
        # 初始化GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # 创建窗口
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(
            self.window_width, 
            self.window_height, 
            "3DGRUT Interactive Viewer", 
            None, 
            None
        )
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        
        # 编译shader程序
        self.shader_program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        
        # 创建四边形网格（用于显示纹理）
        self._setup_quad_vao()
        
        # 创建纹理
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
        # 初始化纹理为白色（用于测试）
        test_texture = np.ones((self.window_height, self.window_width, 3), dtype=np.float32) * 0.5
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.window_width, self.window_height, 
                     0, GL_RGB, GL_FLOAT, test_texture)
        
        # 输入状态
        self.mouse_pos = np.array([0.0, 0.0])
        self.prev_mouse_pos = np.array([0.0, 0.0])
        self.mouse_left_pressed = False
        self.mouse_right_pressed = False
        self.mouse_middle_pressed = False
        self.shift_pressed = False
        self.ctrl_pressed = False
        
        # 相机交互参数（与 Polyscope Turntable 模式完全一致）
        # 参考 Polyscope 官方参数：约 20-25°/秒 @ 60px/秒鼠标速度
        self.camera_rotation_speed = 0.0064     # rad/px (≈ 22.9°/s @ 60px/s = ~15.7秒/360°)
        self.camera_pan_speed = 0.001           # 系数×distance (与Polyscope保持平衡)
        self.camera_zoom_speed = 0.1            # 指数系数（拖动和滚轮缩放，保持一致）
        
        # 初始化相机参数（将由 _compute_scene_bounds 更新）
        self.camera_theta = 0.0  # 水平旋转角
        self.camera_phi = np.pi / 4.0  # 竖直旋转角（初始设为45度）
        
        # 相机姿态复制/粘贴缓存
        self.camera_pose_clipboard = None
        
        # 计算并设置初始相机位置
        self._initialize_camera_from_bounds()
        
        # FPS统计
        
        # FPS统计
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        
        # 渲染参数
        self.render_style_index = 0
        self.render_styles = ["color", "density"]
        
        logger.info("✅ Interactive Viewer initialized successfully!")
        logger.info("\n🎮 Controls (Turntable Mode - 与 Polyscope 一致):")
        logger.info("  Left Mouse Drag           → Rotate camera (旋转相机)")
        logger.info("  Shift + Left Mouse Drag   → Pan camera (平移相机)")
        logger.info("  Right Mouse Drag          → Pan camera (平移相机)")
        logger.info("  Ctrl + Shift + Left Drag  → Zoom in/out (缩放)")
        logger.info("  Scroll Wheel              → Zoom in/out (缩放)")
        logger.info("  S                         → Switch render style (切换渲染样式)")
        logger.info("  R                         → Reset view (重置视图)")
        logger.info("  Ctrl + C                  → Copy camera pose (复制相机姿态)")
        logger.info("  Ctrl + V                  → Paste camera pose (粘贴相机姿态)")
        logger.info("  Q / ESC                   → Quit (退出)")
    
    def _compute_scene_bounds(self):
        """计算场景边界盒（与Polyscope保持一致：固定范围[-1.5, 1.5]³）"""
        try:
            # 获取高斯的位置
            positions = self.engine.scene_mog.positions.detach().cpu().numpy()
            
            # 计算边界盒（用于调试，但不用于相机初始化）
            bbox_min = positions.min(axis=0)
            bbox_max = positions.max(axis=0)
            bbox_size = bbox_max - bbox_min
            scene_diagonal = np.linalg.norm(bbox_size)
            
            # 存储实际场景信息（用于调试）
            self.scene_bbox_min = bbox_min
            self.scene_bbox_max = bbox_max
            self.scene_diagonal = scene_diagonal
            
            # 与 Polyscope 保持一致：固定中心点和范围
            # 假设场景已经经过处理，范围在 [-1.5, 1.5]³ 内
            self.scene_center = np.array([0.0, 0.0, 0.0])      # 固定中心
            self.scene_bbox_min_fixed = np.array([-1.5, -1.5, -1.5])  # Polyscope固定范围
            self.scene_bbox_max_fixed = np.array([1.5, 1.5, 1.5])
            
            logger.info(f"Scene bounds (actual): min={bbox_min}, max={bbox_max}")
            logger.info(f"Scene bounds (Polyscope fixed): min={self.scene_bbox_min_fixed}, max={self.scene_bbox_max_fixed}")
            logger.info(f"Scene diagonal: {scene_diagonal:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute scene bounds: {e}")
            # 使用 Polyscope 的默认值
            self.scene_center = np.array([0.0, 0.0, 0.0])
            self.scene_bbox_min_fixed = np.array([-1.5, -1.5, -1.5])
            self.scene_bbox_max_fixed = np.array([1.5, 1.5, 1.5])
            self.scene_diagonal = 1.0
    
    def _initialize_camera_from_bounds(self):
        """初始化相机（与Polyscope保持一致：固定中心和范围）"""
        # 与 Polyscope 一致：固定中心点和相机距离
        # Polyscope 使用固定的 [-1.5, 1.5]³ 范围
        
        # 场景中心（固定）
        self.camera_pan_x = 0.0  # 固定中心
        self.camera_pan_y = 0.0
        self.camera_pan_z = 0.0
        
        # 计算合适的相机距离，使 [-1.5, 1.5]³ 范围完全可见
        # 对于固定的3×3×3立方体
        bbox_size = 3.0  # [-1.5, 1.5] → 宽度为3
        fov_rad = np.deg2rad(self.engine.camera_fov)  # 45度
        
        # 距离公式：size / (2 * tan(FOV/2)) * 1.0（Polyscope的标准做法）
        # tan(22.5°) ≈ 0.414
        self.camera_distance = bbox_size / (2.0 * np.tan(fov_rad / 2.0))
        # 45° FOV: distance = 3.0 / (2 * tan(22.5°)) ≈ 3.0 / 0.828 ≈ 3.625
        
        # 初始化旋转角度 - 与 Polyscope Free 模式的初始视角类似
        # 但由于我们限制在 Turntable 模式，设置为好的观察角度
        self.camera_theta = 0.0      # 从+X方向看
        self.camera_phi = np.pi / 4.0  # 45度（而不是30度，更接近Polyscope的Free模式）
        
        # 更新相机矩阵
        self._update_camera()
        
        logger.info(f"Camera initialized (Polyscope-like mode):")
        logger.info(f"  Fixed center: ({self.camera_pan_x:.1f}, {self.camera_pan_y:.1f}, {self.camera_pan_z:.1f})")
        logger.info(f"  Fixed range: [-1.5, 1.5]³")
        logger.info(f"  Camera distance: {self.camera_distance:.4f}")
        logger.info(f"  Initial theta: {np.degrees(self.camera_theta):.1f}°, phi: {np.degrees(self.camera_phi):.1f}°")
    
    def _update_camera(self):
        """更新相机参数"""
        # 计算相机位置（球面坐标）
        cam_x = self.camera_distance * np.sin(self.camera_phi) * np.cos(self.camera_theta)
        cam_y = self.camera_distance * np.cos(self.camera_phi)
        cam_z = self.camera_distance * np.sin(self.camera_phi) * np.sin(self.camera_theta)
        
        # 应用平移（围绕场景中心）
        eye = np.array([cam_x + self.camera_pan_x, cam_y + self.camera_pan_y, cam_z + self.camera_pan_z], dtype=np.float64)
        at = np.array([self.camera_pan_x, self.camera_pan_y, self.camera_pan_z], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        
        # 构建标准 LookAt 矩阵
        # 参考 OpenGL 的标准 gluLookAt 实现
        # forward: 从 eye 指向 at
        forward = at - eye
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            forward = np.array([0.0, 0.0, -1.0])
        else:
            forward = forward / forward_norm
        
        # right 轴：forward × up（叉积）
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # forward 平行于 up，调整 up 向量
            up = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
        right = right / right_norm
        
        # 重新计算 up 以确保正交性
        up_normalized = np.cross(right, forward)
        
        # 构建 World-to-Camera 矩阵（标准格式）
        # [ right.x  right.y  right.z  -dot(right, eye) ]
        # [ up.x     up.y     up.z     -dot(up, eye)     ]
        # [ -forward.x -forward.y -forward.z dot(forward, eye) ]
        # [ 0         0        0        1                    ]
        view_matrix = np.eye(4, dtype=np.float64)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up_normalized
        view_matrix[2, :3] = -forward  # 注意：Z轴取反（OpenGL 约定）
        view_matrix[0, 3] = -np.dot(right, eye)
        view_matrix[1, 3] = -np.dot(up_normalized, eye)
        view_matrix[2, 3] = np.dot(forward, eye)  # 这里是正的（因为 Z 轴已取反）
        
        # 创建Camera对象
        fov_rad = np.deg2rad(self.engine.camera_fov)
        self.camera = Camera.from_args(
            view_matrix=torch.from_numpy(view_matrix).unsqueeze(0).to(self.engine.device),
            fov=fov_rad,
            width=self.window_width,
            height=self.window_height,
            near=0.01,
            far=100.0,
            dtype=torch.float64,
            device=self.engine.device
        )
    
    def _setup_quad_vao(self):
        """创建用于显示纹理的四边形VAO"""
        quad_vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0
        ], dtype=np.float32)
        
        quad_indices = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32)
        
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        self.quad_ebo = glGenBuffers(1)
        
        glBindVertexArray(self.quad_vao)
        
        # VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        
        # EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quad_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)
        
        # 顶点位置属性（位置0，offset 0，2个float）
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        
        # 纹理坐标属性（位置1，offset 8字节，2个float）
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        
        # 解绑VAO（保持EBO绑定状态）
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
        self.quad_index_count = len(quad_indices)
    
    def _framebuffer_size_callback(self, window, width, height):
        """处理窗口大小变化"""
        if width > 0 and height > 0:
            self.window_width = width
            self.window_height = height
            glViewport(0, 0, width, height)
            
            # 重新创建纹理以匹配新的窗口大小
            glDeleteTextures([self.texture])
            self.texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # 初始化纹理
            test_texture = np.ones((height, width, 3), dtype=np.float32) * 0.2
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 
                        0, GL_RGB, GL_FLOAT, test_texture)
            
            # 更新相机以匹配新的宽高比
            self._update_camera()
    
    def _key_callback(self, window, key, scancode, action, mods):
        """处理键盘输入"""
        # 更新修饰键状态
        self.shift_pressed = (mods & glfw.MOD_SHIFT) != 0
        self.ctrl_pressed = (mods & glfw.MOD_CONTROL) != 0
        
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_S:
                self.render_style_index = (self.render_style_index + 1) % len(self.render_styles)
                logger.info(f"Switch to render style: {self.render_styles[self.render_style_index]}")
            elif key == glfw.KEY_C and self.ctrl_pressed:
                # Ctrl + C: 复制相机姿态
                self.camera_pose_clipboard = {
                    'theta': self.camera_theta,
                    'phi': self.camera_phi,
                    'distance': self.camera_distance,
                    'pan_x': self.camera_pan_x,
                    'pan_y': self.camera_pan_y,
                    'pan_z': self.camera_pan_z
                }
                logger.info("📋 Camera pose copied to clipboard")
            elif key == glfw.KEY_V and self.ctrl_pressed:
                # Ctrl + V: 粘贴相机姿态
                if self.camera_pose_clipboard is not None:
                    self.camera_theta = self.camera_pose_clipboard['theta']
                    self.camera_phi = self.camera_pose_clipboard['phi']
                    self.camera_distance = self.camera_pose_clipboard['distance']
                    self.camera_pan_x = self.camera_pose_clipboard['pan_x']
                    self.camera_pan_y = self.camera_pose_clipboard['pan_y']
                    self.camera_pan_z = self.camera_pose_clipboard['pan_z']
                    self._update_camera()
                    logger.info("📋 Camera pose pasted from clipboard")
                else:
                    logger.info("⚠️  No camera pose in clipboard")
            elif key == glfw.KEY_R:
                # R: 重置视图到初始位置
                self._initialize_camera_from_bounds()
                logger.info("🔄 Camera view reset to home position")
    
    def _mouse_button_callback(self, window, button, action, mods):
        """处理鼠标按钮"""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_left_pressed = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_right_pressed = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.mouse_middle_pressed = (action == glfw.PRESS)
        
        # 更新修饰键状态
        self.shift_pressed = (mods & glfw.MOD_SHIFT) != 0
        self.ctrl_pressed = (mods & glfw.MOD_CONTROL) != 0
        
        if action == glfw.PRESS:
            self.prev_mouse_pos = self.mouse_pos.copy()
    
    def _cursor_pos_callback(self, window, xpos, ypos):
        """处理鼠标移动 - 实现与 Polyscope Turntable 模式一致的交互"""
        self.mouse_pos = np.array([xpos, ypos])
        
        if self.mouse_left_pressed or self.mouse_right_pressed or self.mouse_middle_pressed:
            delta = self.mouse_pos - self.prev_mouse_pos
            
            # 限制delta的大小，避免跳跃
            max_delta = 50.0
            delta = np.clip(delta, -max_delta, max_delta)
            
            # Polyscope Turntable 模式交互规则:
            # - 左键拖动: 旋转相机（围绕场景中心）
            # - Shift + 左键拖动: 平移相机（Pan）
            # - 右键拖动: 平移相机（Pan）
            # - Ctrl + Shift + 左键拖动: 缩放（Zoom）
            
            is_pan = (self.shift_pressed and self.mouse_left_pressed) or self.mouse_right_pressed
            is_zoom = self.ctrl_pressed and self.shift_pressed and self.mouse_left_pressed
            is_rotate = self.mouse_left_pressed and not self.shift_pressed and not self.ctrl_pressed
            
            if is_zoom:
                # 缩放：上下拖动控制距离（指数缩放，与Polyscope Turntable模式一致）
                # 系数 0.1：与滚轮缩放系数相同
                # 计算: distance *= exp(-delta_y * 0.1)
                # 例: 50px拖动 → exp(-5) ≈ 0.0067 → distance缩小至1/150 (过于灵敏)
                # 但通常delta会被限制在50以内，所以max_delta=50处理峰值情况
                zoom_factor = np.exp(-delta[1] * self.camera_zoom_speed)
                self.camera_distance *= zoom_factor
                self.camera_distance = np.clip(self.camera_distance, 0.1, 100.0)
                self._update_camera()
                
            elif is_rotate:
                # 旋转相机（Turntable 风格）- 与 Polyscope Turntable 模式参数一致
                # 系数 0.0064 rad/px：约 22.9°/秒 @ 60px/秒鼠标速度，360°耗时 ~15.7秒
                # 与Polyscope的 20-25°/秒 范围完全相符（中点）
                # 水平拖动改变绕垂直轴的旋转（theta）
                # 竖直拖动改变俯仰角（phi）
                self.camera_theta += delta[0] * self.camera_rotation_speed
                self.camera_phi += delta[1] * self.camera_rotation_speed
                
                # 限制竖直旋转范围，避免相机翻转
                self.camera_phi = np.clip(self.camera_phi, 0.01, np.pi - 0.01)
                self._update_camera()
                
            elif is_pan:
                # 平移相机（Pan）- 与 Polyscope Turntable 平移速度一致
                # 系数 0.001 × distance：平移与旋转的比例约为 1:6.4（旋转0.0064/平移0.001）
                # 计算相机坐标系的基向量
                cam_x = self.camera_distance * np.sin(self.camera_phi) * np.cos(self.camera_theta)
                cam_y = self.camera_distance * np.cos(self.camera_phi)
                cam_z = self.camera_distance * np.sin(self.camera_phi) * np.sin(self.camera_theta)
                eye = np.array([cam_x, cam_y, cam_z]) + np.array([self.camera_pan_x, self.camera_pan_y, self.camera_pan_z])
                
                at = np.array([self.camera_pan_x, self.camera_pan_y, self.camera_pan_z])
                up = np.array([0.0, 1.0, 0.0])
                
                forward = at - eye
                forward_norm = np.linalg.norm(forward)
                if forward_norm > 1e-6:
                    forward = forward / forward_norm
                else:
                    forward = np.array([0.0, 0.0, -1.0])
                
                right = np.cross(forward, up)
                right_norm = np.linalg.norm(right)
                if right_norm > 1e-6:
                    right = right / right_norm
                else:
                    right = np.array([1.0, 0.0, 0.0])
                
                up_normalized = np.cross(right, forward)
                
                # 在世界坐标中进行平移
                pan_speed = self.camera_pan_speed * self.camera_distance
                self.camera_pan_x -= right[0] * delta[0] * pan_speed - up_normalized[0] * delta[1] * pan_speed
                self.camera_pan_y -= right[1] * delta[0] * pan_speed - up_normalized[1] * delta[1] * pan_speed
                self.camera_pan_z -= right[2] * delta[0] * pan_speed - up_normalized[2] * delta[1] * pan_speed
                
                self._update_camera()
            
            self.prev_mouse_pos = self.mouse_pos.copy()
    
    def _scroll_callback(self, window, xoffset, yoffset):
        """处理鼠标滚轮 - 与 Polyscope 一致的缩放行为"""
        # 使用对数缩放以获得更平滑的缩放体验
        zoom_factor = np.exp(-yoffset * self.camera_zoom_speed)
        self.camera_distance *= zoom_factor
        self.camera_distance = np.clip(self.camera_distance, 0.1, 100.0)
        self._update_camera()
    
    @torch.no_grad()
    def render_frame(self):
        """渲染一帧图像"""
        # 判断是否为第一次渲染或相机/场景有变化
        is_first_pass = self.engine.is_dirty(self.camera)
        
        # 从引擎获取渲染结果
        outputs = self.engine.render_pass(self.camera, is_first_pass=is_first_pass)
        
        rgb = outputs["rgb"]
        
        # 调试信息（仅在前几帧输出）
        if self.frame_count < 5:
            logger.info(f"Frame {self.frame_count}: is_first_pass={is_first_pass}, RGB shape: {rgb.shape}, min: {rgb.min():.6f}, max: {rgb.max():.6f}")
        
        # 转换为numpy并转移到CPU
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        
        # 确保数据是float32格式
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)
        
        # 确保值在0-1范围内
        rgb = np.clip(rgb, 0.0, 1.0)
        
        # 处理通道数
        if rgb.ndim == 4:  # (batch, height, width, channels)
            rgb = rgb[0]  # 取第一个batch
        
        if rgb.ndim == 3:
            if rgb.shape[2] == 4:  # RGBA -> RGB
                rgb = rgb[:, :, :3]
            elif rgb.shape[2] == 1:  # 灰度 -> RGB
                rgb = np.repeat(rgb, 3, axis=2)
        
        # 确保数据布局连续
        rgb = np.ascontiguousarray(rgb)
        
        # 更新纹理
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, rgb.shape[1], rgb.shape[0], 
                        GL_RGB, GL_FLOAT, rgb)
    
    def display(self):
        """显示当前渲染内容"""
        # 设置清屏颜色
        glClearColor(0.0, 0.0, 0.0, 1.0)
        # 清屏
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 使用shader程序
        glUseProgram(self.shader_program)
        
        # 绑定VAO
        glBindVertexArray(self.quad_vao)
        
        # 绑定纹理
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        
        # 设置uniform
        texture_loc = glGetUniformLocation(self.shader_program, "texture1")
        glUniform1i(texture_loc, 0)
        
        # 绘制
        glDrawElements(GL_TRIANGLES, self.quad_index_count, GL_UNSIGNED_INT, None)
    
    def run(self):
        """主循环"""
        while not glfw.window_should_close(self.window):
            # 渲染帧
            self.render_frame()
            self.display()
            
            # 交换缓冲区
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            
            # 更新FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                logger.info(f"FPS: {self.fps:.1f}")
                self.frame_count = 0
                self.last_time = current_time
        
        # 清理资源
        self.cleanup()
    
    def cleanup(self):
        """清理OpenGL资源"""
        glDeleteBuffers(1, [self.quad_vbo])
        glDeleteBuffers(1, [self.quad_ebo])
        glDeleteVertexArrays(1, [self.quad_vao])
        glDeleteTextures([self.texture])
        glDeleteProgram(self.shader_program)
        glfw.destroy_window(self.window)
        glfw.terminate()
        logger.info("Viewer closed.")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="3DGRUT Interactive Viewer")
    parser.add_argument(
        "--gs_object", 
        type=str, 
        required=True, 
        help="Path of pretrained 3dgrt checkpoint (.pt / .ingp / .ply file)"
    )
    parser.add_argument(
        "--mesh_assets",
        type=str,
        default=None,
        help="Path to folder containing mesh assets (.obj or .glb format)"
    )
    parser.add_argument(
        "--default_gs_config",
        type=str,
        default="apps/colmap_3dgrt.yaml",
        help="Default config for .ingp, .ply, or .pt files"
    )
    parser.add_argument(
        "--envmap_assets",
        type=str,
        default=None,
        help="Path to folder containing .hdr environment maps"
    )
    parser.add_argument(
        "--buffer_mode",
        type=str,
        choices=["host2device", "device2device"],
        default="device2device",
        help="Buffer mode for CUDA-OpenGL interop"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Window width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Window height"
    )
    
    args = parser.parse_args()
    
    viewer = InteractiveViewer(
        gs_object=args.gs_object,
        mesh_assets_folder=args.mesh_assets,
        default_config=args.default_gs_config,
        envmap_assets_folder=args.envmap_assets,
        width=args.width,
        height=args.height,
        buffer_mode=args.buffer_mode
    )
    
    viewer.run()


if __name__ == "__main__":
    main()