from django.db import models

# Create your models here.

class RegisterUser(models.Model):
    names = models.CharField(max_length=40)
    telnos = models.CharField(max_length=50)


    def __str__(self): #장고 관리자 페이지에서 어떻게 보일지 정하는 함수 
        return self.names 
