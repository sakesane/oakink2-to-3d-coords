import pickle
import json
import os
import re
from glob import glob
import numpy as np
import torch
from manotorch_master.manotorch.manolayer import ManoLayer, MANOOutput

def extract_sequence_id(filename):
    """从scene开头的文件名中提取序列号"""
    # 匹配pattern: scene_01__A001++seq__<序列号>__<时间戳>
    pattern = r'scene_\d+__[A-Z]\d+\+\+seq__([a-f0-9]+)__\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def load_pickle_data(pickle_path):
    """加载pickle文件并提取所需数据"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    frame_id_list = data.get('frame_id_list', [])
    obj_list = data.get('obj_list', [])
    raw_mano = data.get('raw_mano', {})
    
    return frame_id_list, obj_list, raw_mano

def extract_mano_data(frame_id, raw_mano):
    if frame_id not in raw_mano:
        return None
    
    frame_mano = raw_mano[frame_id]
    
    mano_data = {
        'rh__pose_coeffs': frame_mano.get('rh__pose_coeffs', None),
        'lh__pose_coeffs': frame_mano.get('lh__pose_coeffs', None),
        'rh__tsl': frame_mano.get('rh__tsl', None),
        'lh__tsl': frame_mano.get('lh__tsl', None),
        'rh__betas': frame_mano.get('rh__betas', None),
        'lh__betas': frame_mano.get('lh__betas', None)
    }

    mano_3d_coords = compute_mano_3d_coordinates(mano_data)
    if mano_3d_coords:
        mano_data['3d_coordinates'] = mano_3d_coords
    
    return mano_data

def compute_mano_3d_coordinates(mano_data):
    """
    Args:
        mano_data: dict
    
    Returns:
        dict
    """
    try:
        # left hand
        lh_mano_layer = ManoLayer(rot_mode='quat', side='left')
        print("After lh model creating")
        lh_pose = mano_data['lh__pose_coeffs']
        lh_shape = mano_data['lh__betas']
        lh_mano_output:MANOOutput = lh_mano_layer(lh_pose, lh_shape)
        print("After lh model forward")
        # right hand
        rh_mano_layer = ManoLayer(rot_mode='quat', side='right')
        rh_pose = mano_data['rh__pose_coeffs']
        rh_shape = mano_data['rh__betas']
        rh_mano_output:MANOOutput = rh_mano_layer(rh_pose, rh_shape)
        print("After rh")

        results = {}
        results['lh_output'] = {
            'vertices': lh_mano_output.verts, # (B, 778, 3), root(center_joint) relative
            'joints': lh_mano_output.joints, # (B, 21, 3), root relative
            'transforms_abs': lh_mano_output.transforms_abs  # (B, 16, 4, 4), root relative
        }
        results['rh_output'] = {
            'vertices': rh_mano_output.verts, # (B, 778, 3), root(center_joint) relative
            'joints': rh_mano_output.joints, # (B, 21, 3), root relative
            'transforms_abs': rh_mano_output.transforms_abs  # (B, 16, 4, 4), root relative
        }
        
        return results
        
    except Exception as e:
        print(f"mano_layer create / forward ERROR: {e}")
        return None

def save_3d_coordinates_to_file(seq, frame_id, mano_data, base_dir="3d_coordinates"):
    """
    将MANO三维坐标数据保存到文件
    
    Args:
        seq: 序列号
        frame_id: 帧ID
        mano_data: 包含三维坐标的MANO数据
        base_dir: 基础目录名
    """
    if not mano_data or '3d_coordinates' not in mano_data:
        print(f"警告: 帧 {frame_id} 没有三维坐标数据可保存")
        return
    
    # 创建目录结构：3d_coordinates/序列号/帧ID/
    seq_dir = os.path.join(base_dir, seq)
    frame_dir = os.path.join(seq_dir, f"frame_{frame_id}")
    os.makedirs(frame_dir, exist_ok=True)
    
    coords_3d = mano_data['3d_coordinates']
    
    # 为右手数据创建文件
    if 'rh_output' in coords_3d:
        rh_output = coords_3d['rh_output']
        
        if 'vertices' in rh_output and rh_output['vertices'] is not None:
            vertices_file = os.path.join(frame_dir, "rh_vertices.npy")
            np.save(vertices_file, rh_output['vertices'].detach().cpu().numpy() if hasattr(rh_output['vertices'], 'detach') else rh_output['vertices'])
            
        if 'joints' in rh_output and rh_output['joints'] is not None:
            joints_file = os.path.join(frame_dir, "rh_joints.npy")
            np.save(joints_file, rh_output['joints'].detach().cpu().numpy() if hasattr(rh_output['joints'], 'detach') else rh_output['joints'])
        
        if 'transforms_abs' in rh_output and rh_output['transforms_abs'] is not None:
            transforms_abs_file = os.path.join(frame_dir, "rh_transforms_abs.npy")
            np.save(transforms_abs_file, rh_output['transforms_abs'].detach().cpu().numpy() if hasattr(rh_output['transforms_abs'], 'detach') else rh_output['transforms_abs'])

    # 为左手数据创建文件
    if 'lh_output' in coords_3d:
        lh_output = coords_3d['lh_output']
        
        if 'vertices' in lh_output and lh_output['vertices'] is not None:
            vertices_file = os.path.join(frame_dir, "lh_vertices.npy")
            np.save(vertices_file, lh_output['vertices'].detach().cpu().numpy() if hasattr(lh_output['vertices'], 'detach') else lh_output['vertices'])
            
        if 'joints' in lh_output and lh_output['joints'] is not None:
            joints_file = os.path.join(frame_dir, "lh_joints.npy")
            np.save(joints_file, lh_output['joints'].detach().cpu().numpy() if hasattr(lh_output['joints'], 'detach') else lh_output['joints'])
            
        if 'transforms_abs' in lh_output and lh_output['transforms_abs'] is not None:
            transforms_abs_file = os.path.join(frame_dir, "lh_transforms_abs.npy")
            np.save(transforms_abs_file, lh_output['transforms_abs'].detach().cpu().numpy() if hasattr(lh_output['transforms_abs'], 'detach') else lh_output['transforms_abs'])
            
    # 创建综合信息文件
    summary_file = os.path.join(frame_dir, "summary.json")
    summary_data = {
        "sequence_id": seq,
        "frame_id": frame_id,
        "timestamp": os.path.getmtime(frame_dir) if os.path.exists(frame_dir) else None,
        "available_data": []
    }
    
    if 'rh_output' in coords_3d:
        rh_info = {"hand": "right"}
        if 'vertices' in coords_3d['rh_output'] and coords_3d['rh_output']['vertices'] is not None:
            rh_info["vertices_shape"] = list(coords_3d['rh_output']['vertices'].shape)
            summary_data["available_data"].append({"type": "vertices", "hand": "right", "file": "rh_vertices.npy"})
        if 'joints' in coords_3d['rh_output'] and coords_3d['rh_output']['joints'] is not None:
            rh_info["joints_shape"] = list(coords_3d['rh_output']['joints'].shape)
            summary_data["available_data"].append({"type": "joints", "hand": "right", "file": "rh_joints.npy"})
        if 'transforms_abs' in coords_3d['rh_output'] and coords_3d['rh_output']['transforms_abs'] is not None:
            rh_info["transforms_abs_shape"] = list(coords_3d['rh_output']['transforms_abs'].shape)
            summary_data["available_data"].append({"type": "transforms_abs", "hand": "right", "file": "rh_transforms_abs.npy"})
    
    if 'lh_output' in coords_3d:
        lh_info = {"hand": "left"}
        if 'vertices' in coords_3d['lh_output'] and coords_3d['lh_output']['vertices'] is not None:
            lh_info["vertices_shape"] = list(coords_3d['lh_output']['vertices'].shape)
            summary_data["available_data"].append({"type": "vertices", "hand": "left", "file": "lh_vertices.npy"})
        if 'joints' in coords_3d['lh_output'] and coords_3d['lh_output']['joints'] is not None:
            lh_info["joints_shape"] = list(coords_3d['lh_output']['joints'].shape)
            summary_data["available_data"].append({"type": "joints", "hand": "left", "file": "lh_joints.npy"})
        if 'transforms_abs' in coords_3d['lh_output'] and coords_3d['lh_output']['transforms_abs'] is not None:
            lh_info["transforms_abs_shape"] = list(coords_3d['lh_output']['transforms_abs'].shape)
            summary_data["available_data"].append({"type": "transforms_abs", "hand": "left", "file": "lh_transforms_abs.npy"})
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"已保存帧 {frame_id} 的三维坐标数据到目录: {frame_dir}")

def main():
    pickle_files = glob("anno_preview/*.pkl")
    
    if not pickle_files:
        print("错误: 未找到任何pickle文件")
        return
    
    for i, file in enumerate(pickle_files):
        seq = extract_sequence_id(os.path.basename(file))
        print(f"{i+1}. {os.path.basename(file)}")
    
    try:
        choice = int(input(f"\n请选择要处理的文件 (1-{len(pickle_files)}): ")) - 1
        if choice < 0 or choice >= len(pickle_files):
            print("选择无效")
            return
    except ValueError:
        print("输入无效")
        return
    
    selected_file = pickle_files[choice]
    seq = extract_sequence_id(os.path.basename(selected_file))
    
    if not seq:
        print("错误: 无法从文件名提取序列号")
        return
    
    print(f"\n处理序列: {seq}")
    
    # 加载pickle数据
    print("正在加载pickle文件...")
    try:
        frame_id_list, obj_list, raw_mano = load_pickle_data(selected_file)
        print(f"成功加载: {len(frame_id_list)} 帧, {len(obj_list)} 个对象, {len(raw_mano)} 个mano记录")
    except Exception as e:
        print(f"加载pickle文件失败: {e}")
        return
   
    # 打印可用帧的详细序号
    if frame_id_list:
        print(f"可用帧序号: {frame_id_list}")
        print(f"总计: {len(frame_id_list)} 帧")
    else:
        print("无可用帧序号")
    
    # 询问用户是否要批量处理所有帧
    batch_choice = input("\n是否要批量处理并保存所有帧的三维坐标数据？(y/n): ").lower().strip()
    if batch_choice == 'y':
        print("开始批量处理...")
        processed_count = 0
        for frame_id in frame_id_list:
            try:
                # 提取mano数据
                mano_data = extract_mano_data(frame_id, raw_mano)
                
                # 保存三维坐标数据到文件
                if mano_data and '3d_coordinates' in mano_data:
                    save_3d_coordinates_to_file(seq, frame_id, mano_data)
                    processed_count += 1
                
                # 每处理100帧显示一次进度
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count} 帧...")
                    
            except Exception as e:
                print(f"处理帧 {frame_id} 时出错: {e}")
                continue
        
        print(f"批量处理完成! 总共处理了 {processed_count} 帧")
        return
    
    while True:
        try:
            user_input = input("\ninput frame ID or 'quit' ")
            if user_input.lower() == 'quit':
                break
            
            frame_id = int(user_input)
            
            if frame_id not in frame_id_list:
                print(f"{frame_id}: invalid frame ID")
                continue
            
            mano_data = extract_mano_data(frame_id, raw_mano)
            
            if mano_data and '3d_coordinates' in mano_data:
                save_3d_coordinates_to_file(seq, frame_id, mano_data)

        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
