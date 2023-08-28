from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import requests

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)


def completions(text, max_len=5, num_seq=5):

    _suggestions = generator(text,
                             max_length=max_len,
                             num_return_sequences=num_seq)
    suggestions = [x["generated_text"] for x in _suggestions]
    return suggestions



# Create your views here.
def index(request):
    return render(request, "editor.html")


@csrf_exempt
def recommend(request):

    url = 'http://127.0.0.1:8000/get_suggestions/'
    query = request.POST
    textarray = query.get("text").split("X2X2CF\n")[1:]
    writing = "".join(textarray).replace("\xa0\n", "").strip()
    query = {'text': writing}
    response = requests.get(url,
                            params=query)
    suggestions_d = dict(response.json())['suggestions']
    suggestions = [x['generated_text'] for x in suggestions_d]
    print("-----------")
    print(suggestions)
    print("-----------")
    return HttpResponse(json.dumps({"suggestions": suggestions}),
                        content_type="application/json")
