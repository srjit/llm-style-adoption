from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json


# Create your views here.

def index(request):
    
    return render(request, "editor.html")

@csrf_exempt
def recommend(request):

    query = request.POST
    textarray = query.get("text").split("X2X2CF\n")[1:]
    note = "".join(textarray).replace("\xa0\n", "").strip()
    
   # suggestions = get_autocompletes(note)
    
    return HttpResponse(json.dumps({"suggestions": ["test completion1", "text completion2", "text completion3", "text completion4", "text completion5"]}), content_type="application/json")
