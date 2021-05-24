conda create -n py27 python=2.7

conda activate py27

pip install ipykernel



python -m ipykernel install --name py27





C:\ProgramData\jupyter\kernels\py27



```shell
PS C:\Users\QiuSir> jupyter kernelspec list
Available kernels:
  python3    C:\dev\anaconda3\share\jupyter\kernels\python3
  py27       C:\ProgramData\jupyter\kernels\py27
PS C:\Users\QiuSir>
```

jupyter kernelspec list

http://graphviz.gitlab.io/download/#windows

安装



conda install graphviz

pip install graphviz



注意，graphviz要下载zip包，解压缩到ANACANDA3中，设置PATH

国内源：
新版ubuntu要求使用https源，要注意。

清华：https://pypi.tuna.tsinghua.edu.cn/simple

阿里云：http://mirrors.aliyun.com/pypi/simple/

中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/

华中理工大学：http://pypi.hustunique.com/

山东理工大学：http://pypi.sdutlinux.org/ 

豆瓣：http://pypi.douban.com/simple/

临时使用：
可以在使用pip的时候加参数-i https://pypi.tuna.tsinghua.edu.cn/simple

例如：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyspider，这样就会从清华这边的镜像去安装pyspider库。


