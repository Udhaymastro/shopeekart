from django.http import HttpResponse,JsonResponse
from django.shortcuts import redirect,render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from shopkart.models import user



@csrf_exempt
def registerform(request):
    # if request.method =='POST':
        print('request')
        return render(request,'registerpage.html')

@csrf_exempt
def loginform(request):
        # if request.method =='POST':
            print('request login')
            return render(request,'loginpage.html')

@csrf_exempt
def signin(request):
 if request.method == 'POST':
        email = request.POST["email"]
        password = request.POST["password"]
        


@csrf_exempt
def signup(request):
        print(request,'signup')
        email = request.POST["email"]
        password = request.POST["password"]
        fullname = request.POST['fullname']
        phonenumber = request.POST['phonenumber']
        
        
