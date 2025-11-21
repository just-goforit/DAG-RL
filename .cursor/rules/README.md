# Cursor Rules 使用指南

本项目包含一套完整的 Cursor Rules，用于提升 AI 辅助开发体验。这些规则会自动帮助 Cursor 理解项目结构、编码规范和最佳实践。

## 📚 规则文件列表

### 1. [project-structure.mdc](project-structure.mdc)
**自动应用** | 项目结构总览

- 项目文件组织说明
- 核心模块介绍
- 开发流程指引
- 关键依赖说明

**何时有用**：导航代码库、理解项目架构

---

### 2. [coding-standards.mdc](coding-standards.mdc)
**应用于 Python 文件** | 编码规范

- Python 类型注解规范
- PyTorch 最佳实践
- Gymnasium 环境开发规范
- 命名约定和代码风格
- 性能优化技巧

**何时有用**：编写新代码、代码审查

---

### 3. [model-architecture.mdc](model-architecture.mdc)
**按需引用** | 模型架构指南

- GNN 特征提取器配置
- SAGEConv vs GATConv 选择
- 策略-价值网络设计
- Transformer 架构支持
- 模型参数调优
- 常用配置示例

**何时有用**：设计/修改模型架构、调试模型性能

---

### 4. [training-workflow.mdc](training-workflow.mdc)
**按需引用** | 训练流程指南

- PPO 超参数详解
- 学习率调度策略
- 奖励塑形技巧
- TensorBoard 监控
- 模型保存和恢复
- 课程学习策略
- 常见问题排查

**何时有用**：训练模型、调整超参数、解决训练问题

---

### 5. [environment-dev.mdc](environment-dev.mdc)
**应用于 env/ 目录** | 环境开发指南

- Red-Blue Pebble Game 机制
- 观测和动作空间设计
- 奖励函数设计
- 约束机制实现
- 调试工具使用
- 性能优化技巧

**何时有用**：修改环境、添加新约束、调试环境行为

---

### 6. [cuda-optimization.mdc](cuda-optimization.mdc)
**应用于 CUDA 文件** | CUDA 和性能优化

- CUDA 扩展编译安装
- 稀疏矩阵乘法优化
- GPU 利用率优化
- 内存优化技巧
- 性能分析工具
- 常见性能问题解决

**何时有用**：性能优化、CUDA 开发、内存问题排查

---

## 🚀 快速开始

### Cursor 会自动应用规则

大多数规则会在你编辑相关文件时自动生效：

1. **自动应用的规则**：
   - `project-structure.mdc` - 始终生效
   - `coding-standards.mdc` - 编辑 `.py` 文件时
   - `environment-dev.mdc` - 编辑 `env/` 目录文件时
   - `cuda-optimization.mdc` - 编辑 CUDA 文件时

2. **按需引用的规则**：
   - 在对话中提及规则主题即可
   - 例如："根据 model architecture 规则，我应该如何配置 GNN？"

### 手动引用规则

你也可以在对话中显式引用规则：

```
@model-architecture 我想使用 Transformer 架构，应该如何配置？
```

```
@training-workflow 训练不收敛，帮我诊断问题
```

## 💡 使用技巧

### 1. 探索新领域

当你需要修改不熟悉的模块时：

```
我想修改环境的奖励函数，根据 environment-dev 规则有哪些最佳实践？
```

### 2. 解决问题

遇到训练或性能问题时：

```
根据 training-workflow 规则，GPU 利用率低应该如何解决？
```

### 3. 代码审查

让 AI 根据规则审查代码：

```
根据 coding-standards 规则审查这段代码
[粘贴代码]
```

### 4. 学习最佳实践

```
根据 model-architecture 规则，GNN 和 MLP 各有什么优缺点？
```

## 📖 规则内容说明

### 文件格式

每个规则文件使用 Markdown 格式（`.mdc` 扩展名），包含：

1. **前置元数据**（YAML frontmatter）：
   ```yaml
   ---
   alwaysApply: true              # 始终应用
   description: "规则描述"         # 规则说明（按需应用）
   globs: *.py,*.tsx              # 文件类型匹配（特定文件应用）
   ---
   ```

2. **规则内容**：Markdown 格式的指南和说明

3. **文件引用**：使用 `[filename](mdc:filename)` 格式引用项目文件

### 引用项目文件

规则中大量使用了文件引用，Cursor 可以快速定位：

```markdown
查看 [main.py](mdc:main.py) 了解训练入口
```

点击这些链接可以直接跳转到对应文件。

## 🔧 自定义规则

### 添加新规则

1. 在 `.cursor/rules/` 目录创建新的 `.mdc` 文件
2. 添加 YAML frontmatter 指定应用方式
3. 编写 Markdown 格式的内容

示例：

```markdown
---
description: 我的自定义规则
---

# 自定义规则标题

规则内容...
```

### 修改现有规则

直接编辑 `.cursor/rules/` 目录中的 `.mdc` 文件即可。

### 禁用规则

重命名或删除不需要的规则文件。

## 📊 规则覆盖范围

| 领域 | 规则文件 | 覆盖内容 |
|-----|---------|---------|
| 项目结构 | project-structure | ✅ 完整 |
| Python 编码 | coding-standards | ✅ 完整 |
| GNN 模型 | model-architecture | ✅ 完整 |
| PPO 训练 | training-workflow | ✅ 完整 |
| 环境开发 | environment-dev | ✅ 完整 |
| CUDA 优化 | cuda-optimization | ✅ 完整 |
| 可视化 | - | ⚠️ 部分（在其他规则中） |
| 测试 | - | ⚠️ 待添加 |

## 🎯 常见使用场景

### 场景1：开始新功能开发

```
我想添加一个新的 GNN 层类型，根据 model-architecture 和 coding-standards 规则，
应该如何设计和实现？
```

### 场景2：性能调优

```
训练速度太慢，根据 cuda-optimization 和 training-workflow 规则，
有哪些优化建议？
```

### 场景3：调试问题

```
环境的动作掩码有问题，根据 environment-dev 规则，
应该如何调试和修复？
```

### 场景4：代码重构

```
根据 coding-standards 规则，帮我重构这个函数以符合项目规范
[粘贴代码]
```

## 📝 维护建议

### 定期更新

随着项目演进，及时更新规则：

1. 添加新的最佳实践
2. 更新配置示例
3. 记录新的常见问题
4. 添加性能基准数据

### 版本控制

规则文件应该：
- ✅ 提交到 Git
- ✅ 在 code review 中审查
- ✅ 编写清晰的提交信息
- ✅ 定期同步到团队

## 🤝 贡献

欢迎改进这些规则！如果你发现：

- 规则中的错误或过时信息
- 缺失的重要最佳实践
- 可以添加的有用示例
- 需要澄清的说明

请直接编辑相应的规则文件。

## 🔗 相关资源

- [Cursor Documentation](https://cursor.sh/docs)
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)

## 📮 反馈

如果这些规则对你有帮助，或者你有改进建议，欢迎反馈！

---

**最后更新**: 2025-11-07
**规则版本**: 1.0
**适用于**: gnn_test 项目

