# SMPL 模型下载指南

## 问题说明

运行代码时出现错误：
```
AssertionError: Path data/smpl/SMPL_NEUTRAL.pkl does not exist!
```

这是因为缺少SMPL模型文件。SMPL模型需要从官方网站单独下载。

## 下载步骤

### 方法1：从SMPL官方网站下载（推荐）

1. **注册账号**
   - 访问：https://smpl.is.tue.mpg.com/
   - 点击 "Register" 注册账号
   - 填写信息并验证邮箱

2. **下载模型**
   - 登录后，访问下载页面
   - 下载 "SMPL for Python" (选择.pkl格式)
   - 文件名通常为：`SMPL_python_v.1.0.0.zip`

3. **解压并放置文件**
   ```bash
   # 解压下载的文件
   unzip SMPL_python_v.1.0.0.zip
   
   # 将模型文件复制到项目目录
   cp SMPL_python_v.1.0.0/smpl/models/*.pkl /home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/data/smpl/
   ```

### 方法2：从SMPL-X仓库下载

1. **访问GitHub**
   - https://github.com/vchoutas/smplx

2. **按照README说明下载**
   - 需要注册并同意条款
   - 下载SMPL和SMPL-X模型

3. **放置文件**
   ```bash
   # SMPL模型
   cp SMPL_{MALE,FEMALE,NEUTRAL}.pkl /home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/data/smpl/
   
   # SMPL-X模型（如果需要）
   cp SMPLX_{MALE,FEMALE,NEUTRAL}.pkl /home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/data/smplx/
   ```

## 需要的文件

### SMPL模型（必需）
放置在 `data/smpl/` 目录：
- `SMPL_MALE.pkl`
- `SMPL_FEMALE.pkl`
- `SMPL_NEUTRAL.pkl`

### SMPL-X模型（可选）
放置在 `data/smplx/` 目录：
- `SMPLX_MALE.pkl`
- `SMPLX_FEMALE.pkl`
- `SMPLX_NEUTRAL.pkl`

## 验证安装

运行以下命令验证文件是否正确放置：

```bash
ls -lh data/smpl/
```

应该看到：
```
-rw-r--r-- 1 user user  xxx SMPL_FEMALE.pkl
-rw-r--r-- 1 user user  xxx SMPL_MALE.pkl
-rw-r--r-- 1 user user  xxx SMPL_NEUTRAL.pkl
-rw-r--r-- 1 user user  xxx smpl_body_parts_2_faces.json
```

## 快速测试

下载完成后，运行：

```bash
python fit_smpl_from_data.py --visualize
```

## 替代方案：使用已有的SMPL模型

如果你已经在其他项目中下载过SMPL模型，可以直接复制：

```bash
# 查找系统中已有的SMPL模型
find ~ -name "SMPL_NEUTRAL.pkl" 2>/dev/null

# 复制到项目目录
cp /path/to/existing/SMPL_*.pkl /home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/data/smpl/
```

## 常见问题

### Q1: 为什么需要单独下载？
**A**: SMPL模型有许可限制，不能直接包含在代码库中分发。

### Q2: 必须下载所有三个性别吗？
**A**: 建议下载全部三个（MALE, FEMALE, NEUTRAL）。如果只下载一个，可以修改代码中的gender参数。

### Q3: 下载速度慢怎么办？
**A**: SMPL官方网站可能在国外，下载速度可能较慢。可以尝试：
- 使用代理
- 从镜像站点下载
- 从学术机构镜像下载

### Q4: 可以使用其他版本的SMPL模型吗？
**A**: 可以，但建议使用最新版本（v1.0.0+）。确保文件格式为.pkl。

## 许可说明

使用SMPL模型需要遵守其许可协议：
- 仅用于非商业研究目的
- 不得重新分发模型文件
- 引用时需注明来源

## 引用

如果你在研究中使用SMPL模型，请引用：

```bibtex
@article{SMPL:2015,
  author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
  title = {{SMPL}: A Skinned Multi-Person Linear Model},
  journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
  month = oct,
  number = {6},
  pages = {248:1--248:16},
  publisher = {ACM},
  volume = {34},
  year = {2015}
}
```

## 下一步

下载完成后，继续运行：

```bash
python fit_smpl_from_data.py --visualize
```

如果仍有问题，请检查：
1. 文件名是否正确（区分大小写）
2. 文件路径是否正确
3. 文件是否损坏（可以检查文件大小）
