#看到镜像

docker image ls -a
docker image ls

# 查看容器

docker ps -a
docker ps

# 搜索镜像

docker search

# 拉取镜像

docker image pull ubuntu

# 从Dockerfile创建镜像

docker build

# 从一个修改的容器创建镜像

docker commit

# 推送镜像

docker push

# 为镜像打标签

docker tag

# 删除镜像

docker rmi




# 创建容器并且启动

docker run -it ubuntu:16.04 /bin/bash

docker run -it $image_id


# 重命名容器

docker rename

# 删除容器，可以删除对应容器，一次可以指定多个容器，如果镜像正在被容器引用则无法删除
docker container rm [id/name]

# 删除容器

docker rm

# 启动、停止、重启容器

docker start $container_id
docker stop $container_id
docker restart $container_id


# 运行 -i 表示interactive交互式，-t 表示得到一个 terminal
docker exec -it $container_id bash
docker exec -it $container_id /bin/bash
exit


# cp到本地
docker cp h55_mini_hdfs_gch:/home/top_battle/000023_0 /home/zhangxinxin/

hdfs dfs -get hdfs:///top_battle/ /home/


argon2:$argon2id$v=19$m=10240,t=10,p=8$z78QyWQVoPS4lxGgpEqUBw$Zos53VmEhvPI4rZn/upKyQ




