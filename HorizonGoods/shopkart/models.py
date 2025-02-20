from django.db import models

# Create your models here.
class user(models.Model):
    id = models.AutoField(primary_key=True,null=False)
    username = models.CharField(max_length=255)
    fullname = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
    phonenumber = models.IntegerField(max_length=10)
    password = models.CharField(max_length=255)
    # def __str__(self):
    #     return self.user
   
