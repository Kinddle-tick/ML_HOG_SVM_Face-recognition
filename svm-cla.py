from sklearn import svm
import numpy as np
import time

len_test = 200
all_img = np.load("all_output.npy")
np.random.shuffle(all_img)
print(f"基本信息：\n 总样本形状：{all_img.shape}\n测试集长度：{len_test}")
choose = np.random.randint(0, len(all_img), size=len_test)

atrain=[]
atest=[]
for i in range(10):
    choose=np.arange(int(i*len(all_img)/10),int((i+1)*len(all_img)/10),dtype=np.int)
    test=all_img[choose]
    train = np.delete(all_img, choose, axis=0)

    t = time.time_ns()
    model = svm.SVC(kernel='linear', C=1, gamma='scale', decision_function_shape='ovo')
    model.fit(train[:, :-1], train[:, -1])
    print(f"训练时间：{(time.time_ns()-t)*1e-9}s")
    t = time.time_ns()
    print("训练集精度：")
    tmptrain=model.score(train[:, :-1], train[:, -1])
    atrain.append(tmptrain)
    print(tmptrain)#训练集精度
    print(f"用时：{(time.time_ns()-t)*1e-9}s")
    t = time.time_ns()
    print("测试集精度：")
    tmptest=model.score(test[:, :-1], test[:, -1])
    atest.append(tmptest)
    print(tmptest) #测试集精度
    print(f"用时：{(time.time_ns()-t)*1e-9}s")

print("train")
print(atrain)
print(np.average(atrain))
print("test")
print(atest)
print(np.average(atest))


x = model.predict(np.load("testing_output.npy")[:,:-1])
with open("pred.txt","w") as F:
    for i in x:
        F.write('{}\n'.format(1 if i == 1 else -1))