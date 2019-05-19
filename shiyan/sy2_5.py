class vector():
    def __init__(self,vector):
        self.setVector(vector)
    def setVector(self,vector):
        if not isinstance(vector,list):
            print('vector must be a list')
            return
        self.__vector=vector
    def sum(self,number):
        if type(number)==list and len(number)==len(self.__vector):
            for i in range(len(number)):
                self.__vector[i]=number[i]+self.__vector[i]
            self.show()
        else :
            print('error: the input must be {} long list'.format(len(self.__vector)))
    def sub(self,number):
        if type(number)==list and len(number)==len(self.__vector):
            for i in range(len(number)):
                self.__vector[i]= self.__vector[i]-number[i]
            self.show()
        else :
            print('error: the input must be {} long list'.format(len(self.__vector)))
    def ride(self,number):
        if type(number)==int:
            for i in range(len(self.__vector)):
                self.__vector[i]=self.__vector[i]*number
            self.show()
        else :
            print('error: the input must be one int')
    def divide(self,number):
        if type(number)==int:
            for i in range(len(self.__vector)):
                self.__vector[i]=self.__vector[i]/number
            self.show()
        else :
            print('error: the input must be one int')
    def show(self):
        print(self.__vector)
v1=vector([1,2,2,1,3])
# v1.show()
v1.sum([2,1,3,2,1])
# v1.show()
v1.sub([2,1,3,2,1])
# v1.show()
v1.ride(2)
# v1.show()
v1.divide(2)
# v1.show()