import pickle
import sys
import argparse

def read_pkl_file(filename):
    try:
        # 读取.pkl文件（二进制只读模式）
        with open(filename, "rb") as f:
            all_data = pickle.load(f)

        # 显示数据基本信息
        print("=== 数据基本信息 ===")
        print(f"数据类型：{type(all_data)}")
        if isinstance(all_data, (list, tuple)):
            print(f"数据长度（元素个数）：{len(all_data)}")
        elif isinstance(all_data, dict):
            print(f"数据长度（键值对个数）：{len(all_data)}")
        else:
            try:
                print(f"数据大小：{len(all_data)}")
            except:
                print("无法获取数据长度")

        # 打印所有数据
        print("\n=== 所有数据 ===")
        print(all_data)
        
    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 不存在")
        sys.exit(1)
    except pickle.UnpicklingError:
        print(f"错误：文件 '{filename}' 不是有效的pkl文件")
        sys.exit(1)
    except Exception as e:
        print(f"处理文件时发生错误：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 设置命令行参数解析（使用--filename作为参数名）
    parser = argparse.ArgumentParser(description='读取并显示pkl文件内容')
    # 添加--filename参数，设置为必须提供
    parser.add_argument('--filename', required=True, help='要读取的pkl文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 读取并显示文件内容
    read_pkl_file(args.filename)
    
