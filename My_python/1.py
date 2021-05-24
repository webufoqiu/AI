import matplotlib.pyplot as plt


# 生成画布和axes对象
# nrows=1和ncols=2分别代表1行和两列
fig,ax = plt.subplots(nrows=1,ncols=2)
print(fig)
print(ax)
ax[0].plot([1,2,3],[4,5,6])
ax[1].scatter([1,2,3],[4,5,6])
plt.show()
