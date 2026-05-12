---
name: 修改项目依赖，更新 Docker 环境。
description: "适用于改了package.json\npackage-lock.json\nrequirements.txt\npyproject.toml\ngo.mod\npom.xml"
---

我刚修改了项目依赖文件，请帮我更新 Docker 环境。

要求：
1. 重新构建相关镜像。
2. 重新启动服务。
3. 检查依赖是否正确安装。
4. 查看日志确认项目是否正常启动。
5. 如果有依赖安装错误，请修复后重新构建。