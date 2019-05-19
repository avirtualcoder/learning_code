class Person():
    def __init__(self,name='',age=20,sex='man'):
        self.setName(name)
        self.setAge(age)
        self.setSex(sex)
    def setName(self,name):
        if not isinstance(name,str):
            print('name must be string')
            return
        self.__name=name
    def setAge(self,age):
        if not isinstance(age,int):
            print('age must be number')
            return
        self.__age=age
    def setSex(self,sex):
        if not isinstance(sex,str):
            print('sex must be string ')
            return
        self.__sex=sex
    def show(self):
        print(self.__name)
        print(self.__age)
        print(self.__sex)



class Student(Person):
    def __init__(self,name='',age='20',sex='man',professional=' '):
        self.setProfessional(professional)
        Person.__init__(self,name,age,sex)
        # self.setName(name)
        # self.setAge(age)
        # self.setSex(sex)

    def setProfessional(self,professional):
        if not isinstance(professional,str):
            print('professional must be string ')
            return
        self.__professional=professional
    def show(self):
        print(self.__professional)
        Person.show(self)


student1=Student('lhj',20,'man','computer')
student1.show()

