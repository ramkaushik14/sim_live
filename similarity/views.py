from django.shortcuts import render
from django.http import HttpResponse
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Create your views here.

def home(request):

    model = joblib.load('sent_model.sav')
    df = pd.read_csv('Precily_Text_Similarity.csv')
    
    sentences = [df['text1'],df['text2']]

    embed_1_df = pd.read_csv("sentence_1_embed.csv")
    embed_2_df = pd.read_csv("sentence_2_embed.csv")

    sentence_1_embed = embed_1_df.to_numpy()
    sentence_2_embed = embed_2_df.to_numpy()


    
    lis =[]
    for i in range(10):

       
       lis.append('<hr><hr><br><br><h2>Similarity between</h2> <h2>Text1</h2> <br> {} <br><h2>and Text2</h2> <br> {} <br><h2>is {}</h2> '.format(sentences[0][i],sentences[1][i],
       cosine_similarity(sentence_1_embed[i].reshape(1, -1),
       sentence_2_embed[i].reshape(1, -1))[0][0]))
       
    
    return HttpResponse(lis)
