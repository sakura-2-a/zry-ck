# ===== 轴承缺陷检测系统前端镜像（M8 部署交付物） =====
# 多阶段构建：Node 编译 Vue → nginx 托管静态产物 + 反向代理 /api
# 构建（项目根目录）：
#   docker build -t bearing-frontend -f frontend/Dockerfile .
FROM node:20-alpine AS builder
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
# 国内构建慢时换源：--registry=https://registry.npmmirror.com
RUN npm install --registry=https://registry.npmmirror.com
COPY frontend .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
