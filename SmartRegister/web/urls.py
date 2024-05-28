from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('adduser/', views.add_user, name = 'adduser'),
    path('delete_user/', views.delete_user, name='delete_user'),
    path('face_test/',views.face_test, name = 'face_test')
]
