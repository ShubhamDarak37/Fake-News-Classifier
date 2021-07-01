from django.contrib import admin
from django.urls import path, include
from fnclassifier import views 

urlpatterns = [
	path('',views.home, name='home'),
	path('classifynews',views.classify,name='classifynews'),

]