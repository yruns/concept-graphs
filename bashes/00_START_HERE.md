# 🎯 从这里开始！

## ✅ 脚本已创建完成

已为您创建完整的场景图生成流程脚本，所有文件位于：

```
/home/shyue/codebase/concept-graphs/bashes/
```

## 🚀 立即开始（三步）

### 1️⃣ 确保 Ollama 运行

**在新终端运行：**
```bash
ollama serve
```

> 💡 保持这个终端运行，不要关闭

### 2️⃣ 下载模型（如果还没下载）

```bash
ollama pull llama3.2-vision:latest
ollama pull llama3.1:8b
```

### 3️⃣ 运行脚本

```bash
cd /home/shyue/codebase/concept-graphs/bashes
bash run_all.sh
```

## 📦 已创建的文件

### 主要脚本（按执行顺序）

1. **`0_sanity_check.sh`** (可选)
   - 3D 重建检查，验证数据质量
   
2. **`1_extract_2d_segmentation.sh`** ⭐
   - 提取 2D 分割和 CLIP 特征
   - 输出: `$REPLICA_ROOT/room0/gsa_results_none/`
   
3. **`2_build_3d_object_map.sh`** ⭐
   - 构建 3D 对象地图
   - 输出: `$REPLICA_ROOT/room0/pcd_saves/*.pkl.gz`
   
4. **`3_visualize_object_map.sh`** (可选)
   - 可视化 3D 对象地图
   
5. **`4_extract_object_captions.sh`** ⭐
   - 提取物体描述（Ollama Vision）
   - 输出: `$REPLICA_ROOT/room0/sg_cache/cfslam_llava_captions.json`
   
6. **`5_refine_object_captions.sh`** ⭐
   - 细化物体描述（Ollama GPT）
   - 输出: `$REPLICA_ROOT/room0/sg_cache/cfslam_gpt-4_responses/`
   
7. **`6_build_scene_graph.sh`** ⭐
   - 构建场景图（Ollama GPT）
   - 输出: `$REPLICA_ROOT/room0/sg_cache/map/scene_map_cfslam_pruned.pkl.gz`
   
8. **`7_visualize_scene_graph.sh`** ⭐
   - 可视化最终场景图

### 辅助脚本

- **`1b_extract_2d_segmentation_detect.sh`**
  - 类别感知的分割模式（可选替代步骤 1）

- **`run_all.sh`** 🎯
  - 一键运行所有必需步骤

### 文档

- **`README.md`** - 完整文档
- **`QUICK_REFERENCE.md`** - 快速参考
- **`00_START_HERE.md`** - 本文档

## 📊 完整流程图

```
RGB-D 图像
    ↓
[步骤 1] 2D 分割 (SAM + CLIP)
    ↓
[步骤 2] 3D 对象地图
    ↓
[步骤 4] 物体描述 (Ollama Vision)
    ↓
[步骤 5] 细化描述 (Ollama GPT)
    ↓
[步骤 6] 构建场景图 (Ollama GPT)
    ↓
[步骤 7] 可视化 ✨
```

## ⏱️ 预计时间

- **总时间**: 35-70 分钟（取决于硬件）
- **场景**: room0（第一个场景）
- **GPU**: 建议 12GB+ 显存

## 📍 输出位置

所有结果保存在：
```
$REPLICA_ROOT/room0/
├── gsa_results_none/           # 2D 分割
├── pcd_saves/                  # 3D 对象地图
└── sg_cache/                   # 场景图
    ├── cfslam_llava_captions.json
    ├── cfslam_gpt-4_responses/
    ├── cfslam_object_relations.json ⭐
    └── map/
        └── scene_map_cfslam_pruned.pkl.gz ⭐
```

## ✅ 运行前检查

```bash
# 1. 检查 Ollama
curl http://localhost:11434/api/tags

# 2. 检查环境变量
source /home/shyue/codebase/concept-graphs/env_vars.bash
echo $REPLICA_ROOT

# 3. 检查数据集
ls $REPLICA_ROOT/room0/results/color/ | head -5
```

## 🎯 推荐运行方式

### 方式 A: 完全自动（推荐）

```bash
cd /home/shyue/codebase/concept-graphs/bashes
bash run_all.sh
```

- ✅ 一键运行所有步骤
- ✅ 自动错误检查
- ✅ 显示进度信息
- ✅ 最终自动可视化

### 方式 B: 逐步运行

```bash
cd /home/shyue/codebase/concept-graphs/bashes

bash 1_extract_2d_segmentation.sh
bash 2_build_3d_object_map.sh
bash 4_extract_object_captions.sh
bash 5_refine_object_captions.sh
bash 6_build_scene_graph.sh
bash 7_visualize_scene_graph.sh
```

- ✅ 更好的控制
- ✅ 可以检查中间结果
- ✅ 出错可以单独重跑

## 🎨 可视化操作

在 Open3D 窗口中：

| 按键 | 功能 |
|------|------|
| **g** | 显示/隐藏场景图 ⭐ 最重要！ |
| **r** | RGB 颜色 |
| **i** | 实例 ID |
| **c** | 类别 |
| **+** | 增大点 |
| **-** | 减小点 |
| **ESC** | 退出 |

## 🐛 如果出错

### Ollama 未运行
```bash
# 在新终端
ollama serve
```

### 模型缺失
```bash
ollama list
ollama pull llama3.2-vision:latest
ollama pull llama3.1:8b
```

### 环境变量问题
```bash
source /home/shyue/codebase/concept-graphs/env_vars.bash
```

### 数据集不存在
```bash
# 检查路径
echo $REPLICA_ROOT
ls $REPLICA_ROOT/room0/
```

## 📚 更多帮助

| 文档 | 用途 |
|------|------|
| `README.md` | 详细文档 |
| `QUICK_REFERENCE.md` | 快速参考 |
| `../README_OLLAMA_CN.md` | 完整中文指南 |
| `../QUICKSTART_OLLAMA.md` | 快速开始指南 |

## 💡 提示

1. **首次运行较慢** - Ollama 加载模型需要时间
2. **保持 Ollama 运行** - 避免重复启动
3. **监控 GPU** - 使用 `nvidia-smi` 查看
4. **检查调试图像** - 在 `sg_cache/cfslam_captions_llava_debug/`
5. **按 g 显示场景图** - 最重要的可视化功能！

## 🎓 学习路径

### 第一次运行
1. ✅ 阅读本文档
2. ✅ 运行 `bash run_all.sh`
3. ✅ 等待完成（30-60 分钟）
4. ✅ 在可视化窗口按 **g** 查看场景图

### 深入了解
1. ✅ 阅读 `README.md`
2. ✅ 逐步运行各个脚本
3. ✅ 检查中间输出
4. ✅ 尝试不同模型

### 高级使用
1. ✅ 修改配置参数
2. ✅ 处理其他场景
3. ✅ 调优模型选择
4. ✅ 批量处理

## ✨ 开始吧！

现在一切就绪，您可以直接运行：

```bash
cd /home/shyue/codebase/concept-graphs/bashes
bash run_all.sh
```

**祝您使用顺利！** 🚀

---

有问题？查看 `README.md` 或 `../README_OLLAMA_CN.md`

