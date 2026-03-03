import gzip
import json
import base64
from pathlib import Path
from typing import Any, Dict

import msgpack
import numpy as np


def decompress_nurec_to_json(nurec_file_path: str, output_json_path: str = None) -> Dict[str, Any]:
    """
    解压 .nurec 文件并转换为 JSON 格式，将二进制数据还原为数组格式。
    
    Args:
        nurec_file_path: .nurec 文件路径
        output_json_path: 输出 JSON 文件路径，如果为 None 则自动生成
    
    Returns:
        解压后的字典数据
    """
    nurec_path = Path(nurec_file_path)
    
    if not nurec_path.exists():
        raise FileNotFoundError(f"文件不存在: {nurec_file_path}")
    
    # 读取 .nurec 文件
    with open(nurec_path, 'rb') as f:
        compressed_data = f.read()
    
    # 先用 gzip 解压
    decompressed_data = gzip.decompress(compressed_data)
    
    # 再用 msgpack 解包
    template = msgpack.unpackb(decompressed_data, raw=False)
    
    # 将二进制数据转换为数组格式
    def convert_bytes_to_arrays(obj: Any, parent_key: str = "") -> Any:
        """递归地将 bytes 对象转换为 numpy 数组，再转换为列表"""
        if isinstance(obj, bytes):
            # 查找对应的 shape 信息
            shape_key = parent_key + ".shape"
            
            # 尝试在 state_dict 中查找 shape
            state_dict = template.get("nre_data", {}).get("state_dict", {})
            shape = state_dict.get(shape_key)
            
            if shape is not None:
                # 根据字段名确定数据类型
                if "n_active_features" in parent_key:
                    dtype = np.int64
                else:
                    # 其他字段默认是 float16（根据模板中的 precision: 16）
                    dtype = np.float16
                
                # 将二进制数据转换为 numpy 数组
                array = np.frombuffer(obj, dtype=dtype)
                
                # 如果有 shape 信息，重塑数组
                if len(shape) > 0:
                    array = array.reshape(shape)
                
                # 转换为 Python 列表
                return array.tolist()
            else:
                # 如果没有 shape 信息，返回 base64 编码（空数组的情况）
                if len(obj) == 0:
                    return []
                # 对于无法确定 shape 的二进制数据，返回 base64
                return {
                    "_type": "bytes",
                    "_data": base64.b64encode(obj).decode('utf-8')
                }
        elif isinstance(obj, dict):
            return {key: convert_bytes_to_arrays(value, f"{parent_key}.{key}" if parent_key else key) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_bytes_to_arrays(item, parent_key) for item in obj]
        else:
            return obj
    
    # 转换二进制数据为数组
    json_serializable = convert_bytes_to_arrays(template)
    
    # 确定输出路径
    if output_json_path is None:
        output_json_path = nurec_path.with_suffix('.json')
    else:
        output_json_path = Path(output_json_path)
    
    # 写入 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"成功解压 {nurec_file_path} 到 {output_json_path}")
    
    return template


# 使用示例
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python decompress_nurec.py <nurec_file_path> [output_json_path]")
        sys.exit(1)
    
    nurec_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        decompress_nurec_to_json(nurec_file, output_file)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)