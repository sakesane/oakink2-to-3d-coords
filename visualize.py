import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from pathlib import Path
import matplotlib
import platform

# 配置中文字体
def setup_chinese_font():
    """配置matplotlib中文字体显示"""
    system = platform.system()
    
    if system == "Windows":
        # Windows系统常见的中文字体
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    elif system == "Darwin":  # macOS
        fonts = ['Heiti TC', 'Arial Unicode MS', 'STHeiti']
    else:  # Linux
        fonts = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    
    for font in fonts:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font]
            matplotlib.rcParams['axes.unicode_minus'] = False
            # 测试字体是否可用
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(fig)
            print(f"使用字体: {font}")
            return
        except:
            continue
    
    print("警告: 未找到合适的中文字体，可能显示乱码")

# 初始化字体设置
setup_chinese_font()

class Hand3DVisualizer:
    def __init__(self, data_dir):
        """
        初始化3D手部数据可视化器
        
        Args:
            data_dir (str): 包含3D坐标数据的目录路径
        """
        self.data_dir = Path(data_dir)
        self.sequence_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
    def load_npy_data(self, npy_file_path):
        """
        加载npy文件数据
        
        Args:
            npy_file_path (str): npy文件路径
            
        Returns:
            numpy.ndarray: 加载的3D坐标数据
        """
        try:
            data = np.load(npy_file_path)
            print(f"加载文件: {npy_file_path}")
            print(f"数据形状: {data.shape}")
            return data
        except Exception as e:
            print(f"加载文件失败 {npy_file_path}: {e}")
            return None
    
    def load_frame_data(self, sequence_id, frame_id, hand_type, data_type):
        """
        加载特定帧的特定数据
        
        Args:
            sequence_id (str): 序列ID
            frame_id (int): 帧ID
            hand_type (str): 手部类型 ("lh" 或 "rh")
            data_type (str): 数据类型 ("joints", "vertices", 或 "tips")
            
        Returns:
            numpy.ndarray or None: 加载的数据
        """
        seq_dir = self.data_dir / sequence_id
        frame_dir = seq_dir / f"frame_{frame_id}"
        data_file = frame_dir / f"{hand_type}_{data_type}.npy"
        
        if not data_file.exists():
            print(f"数据文件不存在: {data_file}")
            return None
        
        return self.load_npy_data(data_file)
    
    def plot_hand_joints(self, joints_data, title="手部关节可视化", hand_type="unknown"):
        """
        可视化手部关节数据
        
        Args:
            joints_data (numpy.ndarray): 关节数据，形状为 (1, 21, 3) 或 (21, 3)
            title (str): 图表标题
            hand_type (str): 手部类型 ("left" 或 "right")
        """
        # 确保数据形状正确
        if joints_data.ndim == 3 and joints_data.shape[0] == 1:
            joints_data = joints_data.squeeze(0)  # 去除第一个维度
        
        if joints_data.shape != (21, 3):
            print(f"警告: 关节数据形状不正确 {joints_data.shape}, 期望 (21, 3)")
            return
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取x, y, z坐标
        x = joints_data[:, 0]
        y = joints_data[:, 1]
        z = joints_data[:, 2]
        
        # 绘制关节点
        scatter = ax.scatter(x, y, z, c=range(21), cmap='viridis', s=100, alpha=0.8)
        
        # 手部关节连接关系 (MANO模型的关节拓扑)
        connections = [
            # 拇指
            [0, 1], [1, 2], [2, 3], [3, 4],
            # 食指
            [0, 5], [5, 6], [6, 7], [7, 8],
            # 中指
            [0, 9], [9, 10], [10, 11], [11, 12],
            # 无名指
            [0, 13], [13, 14], [14, 15], [15, 16],
            # 小指
            [0, 17], [17, 18], [18, 19], [19, 20]
        ]
        
        # 绘制连接线，为不同手指使用不同颜色
        finger_colors = ['red', 'orange', 'yellow', 'green', 'blue']
        finger_ranges = [
            (0, 4),   # 拇指连接
            (4, 8),   # 食指连接
            (8, 12),  # 中指连接
            (12, 16), # 无名指连接
            (16, 20)  # 小指连接
        ]
        
        for i, connection in enumerate(connections):
            start, end = connection
            # 验证关节索引是否有效
            if start >= 21 or end >= 21 or start < 0 or end < 0:
                print(f"警告: 无效的关节连接 [{start}, {end}]")
                continue
                
            # 确定颜色
            color = 'black'  # 默认颜色
            for finger_idx, (range_start, range_end) in enumerate(finger_ranges):
                if range_start <= i < range_end:
                    color = finger_colors[finger_idx]
                    break
            
            ax.plot([x[start], x[end]], 
                   [y[start], y[end]], 
                   [z[start], z[end]], 
                   color=color, alpha=0.7, linewidth=2)
        
        # 添加关节编号
        for i in range(21):
            ax.text(x[i], y[i], z[i], f'{i}', fontsize=8)
        
        # 添加手指颜色图例
        finger_names = ['拇指', '食指', '中指', '无名指', '小指']
        finger_colors = ['red', 'orange', 'yellow', 'green', 'blue']
        legend_elements = []
        for name, color in zip(finger_names, finger_colors):
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=name))
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 设置标签和标题
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title} - {hand_type} hand')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='关节编号')
        
        # 设置相等的坐标轴比例
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()
    
    def plot_hand_vertices(self, vertices_data, title="手部顶点可视化", hand_type="unknown", sample_rate=10):
        """
        可视化手部顶点数据（由于顶点数量很多，可以抽样显示）
        
        Args:
            vertices_data (numpy.ndarray): 顶点数据
            title (str): 图表标题
            hand_type (str): 手部类型
            sample_rate (int): 抽样率，每sample_rate个点显示一个
        """
        # 确保数据形状正确
        if vertices_data.ndim == 3 and vertices_data.shape[0] == 1:
            vertices_data = vertices_data.squeeze(0)
        
        print(f"顶点数据形状: {vertices_data.shape}")
        
        # 抽样显示（顶点数量通常很大）
        sampled_vertices = vertices_data[::sample_rate]
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取x, y, z坐标
        x = sampled_vertices[:, 0]
        y = sampled_vertices[:, 1]
        z = sampled_vertices[:, 2]
        
        # 绘制顶点
        scatter = ax.scatter(x, y, z, c=range(len(x)), cmap='plasma', s=20, alpha=0.6)
        
        # 设置标签和标题
        ax.set_xlabel('X 坐标')
        ax.set_ylabel('Y 坐标')
        ax.set_zlabel('Z 坐标')
        ax.set_title(f'{title} - {hand_type}手 (抽样显示: 每{sample_rate}个点)')
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, label='顶点索引')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_sequence(self, sequence_id=None, frame_id=None):
        """
        可视化指定序列的手部数据
        
        Args:
            sequence_id (str): 序列ID，如果为None则使用第一个找到的序列
            frame_id (int): 帧ID，如果为None则可视化第一个找到的帧
        """
        if not self.sequence_dirs:
            print("未找到任何序列目录")
            return
        
        # 选择序列
        if sequence_id is None:
            seq_dir = self.sequence_dirs[0]
            sequence_id = seq_dir.name
        else:
            seq_dir = self.data_dir / sequence_id
            if not seq_dir.exists():
                print(f"序列目录不存在: {seq_dir}")
                return
        
        print(f"可视化序列: {sequence_id}")
        
        # 查找所有帧目录
        frame_dirs = [d for d in seq_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")]
        
        if not frame_dirs:
            print("未找到任何帧目录")
            return
        
        # 选择帧
        if frame_id is None:
            # 选择第一个可用的帧
            frame_dir = frame_dirs[0]
            frame_id_str = frame_dir.name.replace("frame_", "")
        else:
            frame_dir = seq_dir / f"frame_{frame_id}"
            if not frame_dir.exists():
                print(f"帧目录不存在: {frame_dir}")
                return
            frame_id_str = str(frame_id)
        
        print(f"可视化帧: {frame_id_str}")
        
        # 查找该帧的所有npy文件
        npy_files = list(frame_dir.glob("*.npy"))
        
        if not npy_files:
            print(f"帧 {frame_id_str} 中未找到npy文件")
            return
        
        for npy_file in npy_files:
            print(f"\n处理文件: {npy_file.name}")
            
            # 加载数据
            data = self.load_npy_data(npy_file)
            if data is None:
                continue
            
            # 根据文件名确定数据类型和手部类型
            filename = npy_file.stem
            if "lh" in filename:
                hand_type = "left"
            elif "rh" in filename:
                hand_type = "right"
            else:
                hand_type = "unknown"
            
            if "joints" in filename:
                self.plot_hand_joints(data, f"关节数据 - 帧{frame_id_str} - {filename}", hand_type)
            # elif "vertices" in filename:
            #     self.plot_hand_vertices(data, f"顶点数据 - 帧{frame_id_str} - {filename}", hand_type)
    
    def visualize_all_frames_in_sequence(self, sequence_id=None):
        """
        可视化指定序列中所有帧的数据
        
        Args:
            sequence_id (str): 序列ID，如果为None则使用第一个找到的序列
        """
        if not self.sequence_dirs:
            print("未找到任何序列目录")
            return
        
        # 选择序列
        if sequence_id is None:
            seq_dir = self.sequence_dirs[0]
            sequence_id = seq_dir.name
        else:
            seq_dir = self.data_dir / sequence_id
            if not seq_dir.exists():
                print(f"序列目录不存在: {seq_dir}")
                return
        
        print(f"可视化序列中的所有帧: {sequence_id}")
        
        # 查找所有帧目录
        frame_dirs = [d for d in seq_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")]
        frame_dirs.sort(key=lambda x: int(x.name.replace("frame_", "")))  # 按帧号排序
        
        if not frame_dirs:
            print("未找到任何帧目录")
            return
        
        print(f"找到 {len(frame_dirs)} 个帧目录")
        
        for frame_dir in frame_dirs:
            frame_id = frame_dir.name.replace("frame_", "")
            print(f"\n=== 处理帧 {frame_id} ===")
            
            # 查找该帧的所有npy文件
            npy_files = list(frame_dir.glob("*.npy"))
            
            if not npy_files:
                print(f"帧 {frame_id} 中未找到npy文件")
                continue
            
            for npy_file in npy_files:
                print(f"处理文件: {npy_file.name}")
                
                # 加载数据
                data = self.load_npy_data(npy_file)
                if data is None:
                    continue
                
                # 根据文件名确定数据类型和手部类型
                filename = npy_file.stem
                if "lh" in filename:
                    hand_type = "left"
                elif "rh" in filename:
                    hand_type = "right"
                else:
                    hand_type = "unknown"
                
                if "joints" in filename:
                    self.plot_hand_joints(data, f"Joint Data - frame{frame_id} - {filename}", hand_type)
                # elif "vertices" in filename:
                #     self.plot_hand_vertices(data, f"Vertice Data - frame{frame_id} - {filename}", hand_type)
    
    def list_available_sequences(self):
        """
        列出所有可用的序列
        """
        print("可用的序列:")
        for i, seq_dir in enumerate(self.sequence_dirs):
            print(f"{i+1}. {seq_dir.name}")
            
            # 查找帧目录
            frame_dirs = [d for d in seq_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")]
            print(f"   - 帧目录数量: {len(frame_dirs)}")
            
            if frame_dirs:
                # 统计文件数量
                total_npy_files = 0
                frame_ids = []
                for frame_dir in frame_dirs:
                    npy_files = list(frame_dir.glob("*.npy"))
                    total_npy_files += len(npy_files)
                    frame_id = int(frame_dir.name.replace("frame_", ""))
                    frame_ids.append(frame_id)
                
                frame_ids.sort()
                print(f"   - 总npy文件数量: {total_npy_files}")
                
                # 查看第一个帧的summary文件
                first_frame_dir = seq_dir / f"frame_{frame_ids[0]}"
                summary_file = first_frame_dir / "summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        print(f"   - 示例帧数据: {len(summary.get('available_data', []))} 种数据类型")
                    except:
                        pass
    
    def list_frames_in_sequence(self, sequence_id):
        """
        列出指定序列中的所有帧
        
        Args:
            sequence_id (str): 序列ID
        """
        seq_dir = self.data_dir / sequence_id
        if not seq_dir.exists():
            print(f"序列目录不存在: {seq_dir}")
            return
        
        frame_dirs = [d for d in seq_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")]
        frame_dirs.sort(key=lambda x: int(x.name.replace("frame_", "")))
        
        print(f"序列 {sequence_id} 中的可用帧:")
        for frame_dir in frame_dirs:
            frame_id = frame_dir.name.replace("frame_", "")
            npy_files = list(frame_dir.glob("*.npy"))
            summary_file = frame_dir / "summary.json"
            
            print(f"  帧 {frame_id}:")
            print(f"    - npy文件: {len(npy_files)} 个")
            
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    available_data = summary.get('available_data', [])
                    data_types = {}
                    for data in available_data:
                        hand = data.get('hand', 'unknown')
                        data_type = data.get('type', 'unknown')
                        if hand not in data_types:
                            data_types[hand] = []
                        data_types[hand].append(data_type)
                    
                    for hand, types in data_types.items():
                        print(f"    - {hand}手: {', '.join(types)}")
                except:
                    print(f"    - 无法读取summary文件")
            else:
                print(f"    - 无summary文件")

def main():
    # 设置数据目录
    data_dir = "3d_coordinates"
    
    # 创建可视化器
    visualizer = Hand3DVisualizer(data_dir)
    
    # 列出可用序列
    visualizer.list_available_sequences()
    
    print("\n" + "="*50)
    print("可视化选项:")
    print("1. 可视化单个帧")
    print("2. 可视化序列中的所有帧")
    print("3. 列出序列中的所有帧")
    
    try:
        choice = input("\n请选择操作 (1-3): ").strip()
        
        if choice == "1":
            # 可视化单个帧
            if not visualizer.sequence_dirs:
                print("未找到任何序列目录")
                return
            
            # 选择序列
            if len(visualizer.sequence_dirs) == 1:
                seq_id = visualizer.sequence_dirs[0].name
                print(f"使用序列: {seq_id}")
            else:
                print("\n请选择序列:")
                for i, seq_dir in enumerate(visualizer.sequence_dirs):
                    print(f"{i+1}. {seq_dir.name}")
                seq_choice = int(input("请输入序列编号: ")) - 1
                seq_id = visualizer.sequence_dirs[seq_choice].name
            
            # 列出该序列中的帧
            visualizer.list_frames_in_sequence(seq_id)
            
            # 选择帧
            frame_id = input("\n请输入要可视化的帧ID: ").strip()
            try:
                frame_id = int(frame_id)
                visualizer.visualize_sequence(seq_id, frame_id)
            except ValueError:
                print("帧ID必须是数字")
                
        elif choice == "2":
            # 可视化所有帧
            if not visualizer.sequence_dirs:
                print("未找到任何序列目录")
                return
            
            if len(visualizer.sequence_dirs) == 1:
                seq_id = visualizer.sequence_dirs[0].name
                print(f"使用序列: {seq_id}")
            else:
                print("\n请选择序列:")
                for i, seq_dir in enumerate(visualizer.sequence_dirs):
                    print(f"{i+1}. {seq_dir.name}")
                seq_choice = int(input("请输入序列编号: ")) - 1
                seq_id = visualizer.sequence_dirs[seq_choice].name
            
            visualizer.visualize_all_frames_in_sequence(seq_id)
            
        elif choice == "3":
            # 列出帧
            if not visualizer.sequence_dirs:
                print("未找到任何序列目录")
                return
            
            if len(visualizer.sequence_dirs) == 1:
                seq_id = visualizer.sequence_dirs[0].name
                print(f"使用序列: {seq_id}")
            else:
                print("\n请选择序列:")
                for i, seq_dir in enumerate(visualizer.sequence_dirs):
                    print(f"{i+1}. {seq_dir.name}")
                seq_choice = int(input("请输入序列编号: ")) - 1
                seq_id = visualizer.sequence_dirs[seq_choice].name
            
            visualizer.list_frames_in_sequence(seq_id)
            
        else:
            print("无效的选择")
            
    except (ValueError, IndexError, KeyboardInterrupt) as e:
        print(f"操作被取消或输入无效: {e}")

if __name__ == "__main__":
    main()
