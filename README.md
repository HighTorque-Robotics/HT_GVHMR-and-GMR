**GMR 与 GVHMR 项目整合与使用指南**

概述

本文档介绍如何在同一虚拟环境中整合并运行 GMR 与 GVHMR 项目。这两个项目分别用于人体运动重定向（GMR）和视频人体姿态与形状估计（GVHMR）。

1. 环境安装

为两个项目创建并激活同一个虚拟环境（如 conda 或 venv）。
分别进入 GMR 与 GVHMR 项目目录，根据各自 README.md中的说明安装依赖。

    确保两个项目的依赖无冲突，必要时调整版本。

2. 下载必要的文件

下载人体模型文件、视频识别与骨架提取文件（链接需替换为实际有效链接，此处以 xxxx链接表示）。

    解压后得到 body_models文件夹和 input文件夹。

3. 文件配置

3.1 配置 GMR

    将 body_models文件夹解压，并放置在 GMR/assets/目录下。

3.2 配置 GVHMR

    将 input文件夹解压，并放置在 GVHMR/目录下。

4. 使用流程

4.1 运行 GVHMR

进入 GVHMR/目录。

    按照其 README.md说明，处理视频并提取人体骨架信息。

4.2 运行 GMR

获得视频骨架文件后，进入 GMR/目录。

    根据 args.txt中的参数示例运行 GMR 进行运动重定向。

4.3 BVH 重定向

    关于 GMR 对 bvh 文件重定向的具体方法，请参考 README_unified.md中的说明。

注意事项

确保文件路径正确，尤其是 body_models和 input的放置位置。
运行过程中如遇到依赖问题，请参考各项目的 README.md或 environment.yml进行调试。

