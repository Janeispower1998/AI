# 方法一
str = "{} {}"
print(str.format("Hello", "World"))
# 方法二
str = "{} {}".format("Hello", "World")
print(str)
str1 = "{0}  {1}".format("Hello", "World")
str2 = "{1}  {0}".format("Hello", "World")
str3 = "我村有个傻子叫{0}，{0}是我村的一个傻子".format("小明")
# 不加位置会报错
# str3="我村有个傻子叫{}，{}是我村的一个傻子".format("小明")
print(str1)
print(str2)
print(str3)
