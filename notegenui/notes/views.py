from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json


from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)


def completions(text, max_len=5, num_seq=5):

    _suggestions = generator(text, max_length=max_len, num_return_sequences=num_seq)
    suggestions = [x["generated_text"] for x in _suggestions]

    return suggestions



# Create your views here.

def index(request):
    
    return render(request, "editor.html")

@csrf_exempt
def recommend(request):

    query = request.POST
    textarray = query.get("text").split("X2X2CF\n")[1:]
    note = "".join(textarray).replace("\xa0\n", "").strip()

    suggestions = completions(note)
    print("*****************************************")
    print(suggestions)
    print("*****************************************")
    
   # suggestions = get_autocompletes(note)
    
    return HttpResponse(json.dumps({"suggestions": suggestions}), content_type="application/json")
