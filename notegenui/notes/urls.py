from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path("", views.index, name="index"),
    path("recommend/", views.recommend, name="recommend"),
    path("retrain/", views.retrain, name="retrain"),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
