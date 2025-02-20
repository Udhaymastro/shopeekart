from django.urls import path
from shopkart.views import login

urlpatterns = [
    path('',login.loginform,name='loginform'),
    path('registerform/',login.registerform,name='registerform'),
    path('signin/',login.signin,name='signin'),
    path('signup/',login.signup,name='signup'),
]
