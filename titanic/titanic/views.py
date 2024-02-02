from django.shortcuts import render
from . import fake_model
from . import ml_predict

def home(request):
    return render(request, 'index.html')

def result(request):
    age = int(request.GET["age"])
    sex = int(request.GET["sex"])
    sibsp = int(request.GET["sibsp"])
    pclass = int(request.GET["pclass"])
    parch = int(request.GET["parch"])
    fare = int(request.GET["fare"])
    embarked = int(request.GET["embarked"])
    title = int(request.GET["title"])
    # predication = fake_model.fake_predict(user_input_age)
    prediction = ml_predict.prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title)
    return render(request, 'result.html', {"prediction": prediction})
