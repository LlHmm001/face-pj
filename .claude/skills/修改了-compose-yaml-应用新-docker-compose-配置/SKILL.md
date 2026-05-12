---
name: 修改了 compose.yaml，应用新 Docker Compose 配置。
description: 我刚修改了 compose.yaml，请帮我应用新的 Docker Compose 配置。
---

我刚修改了 compose.yaml，请帮我应用新的 Docker Compose 配置。

要求：
1. 检查 compose.yaml 语法是否正确。
2. 重新创建受影响的容器。
3. 如果涉及镜像变化，请重新 build。
4. 启动所有服务。
5. 检查端口、环境变量、volume 是否生效。
6. 如果失败，请查看日志并修复。