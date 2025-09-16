# class Displayer:
#     def display(self, message):
#         print(message)
#         # self.log()
#         print("World")
#
#     def log(self):
#         print("log")
#
#
# class LoggerMixin:
#     def log(self):
#         super().display("Hello")
#         print("Mixin-log")
#
#
# class MySubClass(LoggerMixin, Displayer):
#     pass
#
# ms = MySubClass()
# print(ms.__mro__)
# ms.display("This string will be shown and logged in subclasslog.txt")
# print('-------------------')
# ms.log()  # 此方法中的super().display("Hello")，执行Displayer类中的display()函数


class A:
    def p(self):
        print('A')
class B():
    def p(self):
        print('B')
class C(A,B):
    def p(self):
        print('C')
class D(C):
    def p(self):
        print('D')
a = A()
b = B()
c = C()
d = D()
print('-----------------')
print(D.__mro__)
super(C, d).p()