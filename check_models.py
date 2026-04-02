import os
import sys

def check_smpl_models():
    smpl_dir = "data/smpl"
    smplx_dir = "data/smplx"
    
    required_smpl_files = [
        "SMPL_MALE.pkl",
        "SMPL_FEMALE.pkl",
        "SMPL_NEUTRAL.pkl"
    ]
    
    required_smplx_files = [
        "SMPLX_MALE.pkl",
        "SMPLX_FEMALE.pkl",
        "SMPLX_NEUTRAL.pkl"
    ]
    
    print("=" * 60)
    print("SMPL 模型文件检查")
    print("=" * 60)
    
    smpl_missing = []
    smpl_found = []
    
    print("\n检查 SMPL 模型:")
    print("-" * 60)
    for filename in required_smpl_files:
        filepath = os.path.join(smpl_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filename:20s} - 存在 ({size_mb:.2f} MB)")
            smpl_found.append(filename)
        else:
            print(f"✗ {filename:20s} - 缺失")
            smpl_missing.append(filename)
    
    smplx_missing = []
    smplx_found = []
    
    print("\n检查 SMPL-X 模型 (可选):")
    print("-" * 60)
    for filename in required_smplx_files:
        filepath = os.path.join(smplx_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filename:20s} - 存在 ({size_mb:.2f} MB)")
            smplx_found.append(filename)
        else:
            print(f"✗ {filename:20s} - 缺失")
            smplx_missing.append(filename)
    
    print("\n" + "=" * 60)
    print("检查结果:")
    print("=" * 60)
    
    if len(smpl_missing) == 0:
        print("✓ SMPL 模型文件完整")
    else:
        print(f"✗ 缺少 {len(smpl_missing)} 个 SMPL 模型文件")
        print("\n缺少的文件:")
        for filename in smpl_missing:
            print(f"  - {filename}")
    
    if len(smplx_missing) == 0:
        print("✓ SMPL-X 模型文件完整")
    elif len(smplx_found) > 0:
        print(f"⚠ SMPL-X 模型不完整 (缺少 {len(smplx_missing)} 个)")
    else:
        print("⚠ SMPL-X 模型未安装 (可选)")
    
    print("\n" + "=" * 60)
    
    if len(smpl_missing) > 0:
        print("\n请下载缺失的 SMPL 模型文件:")
        print("\n方法1: 从官方网站下载")
        print("  1. 访问: https://smpl.is.tue.mpg.com/")
        print("  2. 注册并登录")
        print("  3. 下载 SMPL for Python (.pkl格式)")
        print("  4. 解压并将 .pkl 文件复制到 data/smpl/ 目录")
        
        print("\n方法2: 从 GitHub 仓库下载")
        print("  1. 访问: https://github.com/vchoutas/smplx")
        print("  2. 按照说明下载模型")
        print("  3. 将文件复制到相应目录")
        
        print("\n详细说明请查看: DOWNLOAD_SMPL.md")
        
        return False
    else:
        print("\n✓ 所有必需的模型文件都已安装!")
        print("可以运行: python fit_smpl_from_data.py --visualize")
        return True


if __name__ == "__main__":
    success = check_smpl_models()
    sys.exit(0 if success else 1)
