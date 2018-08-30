import numpy as np

for letter in 'Python':  # 第一个实例
    print('当前字母 :', letter)

fruits = ['banana', 'apple', 'mango']
for fruit in fruits:  # 第二个实例
    print('当前水果 :', fruit)

print("Good bye!")

# 当前字母 : P
# 当前字母 : y
# 当前字母 : t
# 当前字母 : h
# 当前字母 : o
# 当前字母 : n
# 当前水果 : banana
# 当前水果 : apple
# 当前水果 : mango
# Good bye!
################################
x = np.array([[[0], [1], [2]]])
print(x.shape)
print(np.squeeze(x).shape)
print(np.squeeze(x, axis=(2,)).shape)

# # (1, 3, 1)
# # (3,)
# # (1, 3)

#######################################
uid_to_human = {}
seq = [' one ', ' two ', ' three ']
for i, element in enumerate(seq):
    print(i, element)
    uid_to_human[i] = element
    print(uid_to_human[i])

# i = 0
# seq = [' one ', ' two ', ' three ']
# for element in seq:
#     print(i, seq[i])
#     i += 1
######################################################