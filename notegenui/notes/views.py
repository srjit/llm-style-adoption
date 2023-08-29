from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import requests
from urllib.parse import urljoin


API_SERVER = settings.API_SERVER


# Create your views here.
def index(request):
    return render(request, "dashboard.html")


@csrf_exempt
def recommend(request):

    """
    Returns a list of recommendations for a given text
    """
    api_url = urljoin(API_SERVER, 'get_suggestions')

    query = request.POST
    textarray = query.get("text").split("X2X2CF\n")[1:]
    writing = "".join(textarray).replace("\xa0\n", "").strip()
    query = {'text': writing}
    response = requests.get(api_url,
                            params=query)
    suggestions_d = dict(response.json())['suggestions']
    suggestions = [x['generated_text'] for x in suggestions_d]

    return HttpResponse(json.dumps({"suggestions": suggestions}),
                        content_type="application/json")


@csrf_exempt
def retrain(request):
    """
    Build an updated version of the autocomplete model

    """
    query = request.POST
    path = query.get("path")
    api_url = urljoin(API_SERVER, 'retrain')
    query = {'notes_path': path}
    response = requests.get(api_url,
                            params=query)
    return HttpResponse(json.dumps({"status": response}),
                        content_type="application/json")
