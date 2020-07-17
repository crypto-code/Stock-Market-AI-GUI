from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, Http404, JsonResponse
from django.urls import reverse
from Stock.backend.stockinfo import *
from Stock.backend.predict import *
from Stock.backend.trade import *
import sys
import time
# Create your views here.

def index(request):
    return render(request, "index.html")

def stock(request):
    data = None
    stock = None
    if request.method == "POST":
        sym = request.POST["symbol"]
        time = request.POST["period"]
        if time=="1d":
            data = stock_today(sym)
        else:
            data = get_stock(sym, time)
        stock = get_info(sym)
    context = {
        "data": data,
        "symbols": all_symbols,
        "stock": stock
    }
    return render(request, "stock.html", context)

def stock_predict(request, symbol, period, sim, future):
    data = predict_stock(symbol, period, sim, future)
    return JsonResponse({"data": data})

def predict(request):
    data = None
    sym = ""
    if request.method == "POST":
        sym = request.POST["symbol"]
        data = request.POST["plot"]
    
    context = {
        "symbols": all_symbols,
        "data": data,
        "sym": sym
    }
    return render(request, "predict.html", context)

def stock_trade(request, symbol, period, init, skip):
    data = trade_stock(symbol, period, init, skip)
    time.sleep(5)
    return JsonResponse({"data": data})

def trade(request):
    data = None
    sym = ""
    if request.method == "POST":
        sym = request.POST["symbol"]
        data = request.POST["plot"]

    context = {
        "symbols": all_symbols,
        "data": data,
        "sym": sym
    }
    return render(request, "trade.html", context)