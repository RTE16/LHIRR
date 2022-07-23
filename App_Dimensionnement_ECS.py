# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:25:38 2022

@author: RomaneTEZE
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from datetime import timedelta
import plotly.express as px
from math import pi
from math import sqrt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

st.set_page_config(layout = "wide")
st.markdown('# Dimensionnement - Production ECS')
st.sidebar.header("Profil de puisage")
st.markdown('## Profil de puisage')
uploaded_file = st.sidebar.file_uploader("Sélectionnez le fichier", type = ['csv'])
st.sidebar.number_input("Entrez la fréquence des données", min_value=1, value=60, step=1, key="freq")

#st.session_state.chemin
@st.cache
def jour(x):
    """
    fonction qui extrait le jour d'une date complète
    retourne 0 pour lundi, 1 pour mardi etc
    """
    return datetime.strptime(str(x),'%d/%m/%Y').weekday()

@st.cache
def profil_puisage(chemin):
    """
    lecture du fichier csv des données de puisage à partir d'un chemin d'accès à spécifier en entrée
    On impose que le fichier csv d'entrée possède les colonnes : Dates courtes, UT time, et 
    volume d'eau chaude puisé en litres
    
    Le fichier csv contient les données sur une semaine, heure par heure 
        
    le jour de la semaine sera donné par .weekday() : lundi = 0, mardi = 1 ... dimanche=6
    """
   
    df=pd.read_csv(chemin,sep=";", decimal=',')
    table = df[['Date','UT time','Volume puise']]
    colonne_jours=pd.DataFrame(len(table)*[['a']],columns=['Jour'])
    table=pd.concat([table,colonne_jours],axis=1)
    table['Jour']=table['Date'].apply(jour)
    #table['Date']=pd.to_datetime(table['Date'], dayfirst=True)
    #table['UT time'] = pd.to_datetime(table['UT time'])
    table['Date'] = pd.to_datetime(table['Date'] + " " + table['UT time'], dayfirst=True)
    #table['UT time']=pd.date_range(table['Date'].iloc[0],table['Date'].iloc[-1]+timedelta(hours=23)+timedelta(minutes=(60-pas)),freq=str(pas/60)+'H')
    #table['Date']=table['UT time']
    del table['UT time']

    return table

tab_puisage = profil_puisage(uploaded_file)

if st.checkbox('Montrer les données de puisage'):
    st.markdown('> **Données de puisage**')
    gb = GridOptionsBuilder.from_dataframe(tab_puisage)
    gb.configure_pagination()
    gridOptions = gb.build()
    AgGrid(tab_puisage, gridOptions=gridOptions, fit_columns_on_grid_load=True)
    
dt = tab_puisage['Date']

def afficher_profil(profil):
    " Profil est un dataframe contenant les dates et les volumes puisés en litres "
    
    fig = px.bar(profil, x='Date', y = 'Volume puise', 
                     labels = {'Date' : 'Date', 'Volume puise' : 'Volume puisé (Litres)'}, 
                     title = 'Profil de puisage')
    
    fig.update_xaxes(rangeslider_visible=True)
    
    #fig.show()
    
    return fig

fig = afficher_profil(tab_puisage)

st.plotly_chart(fig, use_container_width=True)

st.sidebar.header("Consignes de températures")
st.markdown("## Consignes de températures")
st.sidebar.number_input("Entrez la température de production", min_value=0.0, value = 60.0, step=0.1, key="Tprod")
st.sidebar.number_input("Entrez la température d'eau froide", min_value=0.0, value=10.0, step=0.1, key="Tfroid")
st.sidebar.number_input("Entrez la température de puisage", min_value=0.0, value=55.0, step=0.1, key="Tpuis")
st.sidebar.number_input("Entrez la température de stockage", min_value=0.0, value=60.0, step=0.1, key="Tstock")
st.sidebar.number_input("Entrez la température ambiante", min_value=0.0, value=18.0, step=0.1, key="Tamb")
st.sidebar.number_input("Entrez la température ambiante au primaire", min_value=0.0, value=13.0, step=0.1, key="Tambprim")

def display_temp(dt, temp_prod, temp_froid, temp_puis, **kwargs) :
    temp_prod = pd.Series(data=int(temp_prod), index=dt.index, name="Température")
    temp_froid = pd.Series(data=int(temp_froid), index=dt.index, name="Température")
    temp_puis=pd.Series(data=int(temp_puis), index=dt.index, name="Température")
    
    df1 = pd.concat([temp_prod, dt], axis=1)
    df2 = pd.concat([temp_froid, dt], axis=1)
    df3 = pd.concat([temp_puis, dt], axis=1)
    
    df1['Type'] = 'Production'
    df2['Type'] = 'Froid'
    df3['Type'] = 'Puisage'
    
    df=pd.concat([df1, df2, df3], axis=0)
    
    for key, value in kwargs.items() :
        if key=='temp_amb' :
            temp_amb = pd.Series(data=int(value), index=dt.index, name="Température")
            df5 = pd.concat([temp_amb,dt], axis=1)
            df5['Type'] = 'Ambiante'
            df = pd.concat([df, df5], axis=0)
        if key=='temp_retour' :
            temp_retour = pd.Series(data=int(value), index=dt.index, name="Température")
            df6 = pd.concat([temp_retour, dt], axis=1)
            df6['Type'] = 'Retour'
            df = pd.concat([df, df6], axis=0)
        if key=='temp_stock' :
            temp_stock = pd.Series(data=int(value), index=dt.index, name="Température")
            df7 = pd.concat([temp_stock, dt], axis=1)
            df7['Type'] = 'Stockage'
            df = pd.concat([df, df7], axis=0)
        
    
    fig = px.scatter(df, x='Date', y ='Température', color='Type',
                     labels = { 'Date' : 'Date',
                               'Température' : 'Température, °C'},
                     #color_discrete_map={"Froid" : "blue", "Production" : "red", 
                      #                        "Puisage" : "Purple", 'Stockage' : "yellow", "Ambiante" : "green"},
                     title = 'Températures en °C', color_discrete_sequence=px.colors.qualitative.Set1)
    
    fig.update_xaxes(rangeslider_visible=True)
    
    #fig.show()
    
    return fig

fig2 = display_temp(dt, st.session_state.Tprod, st.session_state.Tfroid, st.session_state.Tpuis, temp_stock = st.session_state.Tstock, temp_amb = st.session_state.Tamb)
st.plotly_chart(fig2, use_container_width=True)

st.sidebar.header("Coefficient d'efficacité du stockage")
st.sidebar.number_input("Entrez le coefficient", min_value=0.01, max_value=1.0, value=1.0, step=0.01, key="coeff_stock")

st.markdown("## Volumes équivalents")

@st.cache
def equivalence(profil, temp_puis, temp_prod, temp_froid):
    ''' Fonction qui ajoute au profil une Series de volume équivalent, dans le cas où la Series de volumes
    puisés  correspond aux volumes puisés à une température inférieure à la température de production (par ex,
    en cas de présence d'un mitigeur) '''
    
    if temp_puis>temp_prod :
        print("Erreur ! La température de puisage est supérieure à la température d'eau chaude")
        return None
    
    volume_eq = pd.Series(index=profil.index, data=profil['Volume puise']*((temp_puis-temp_froid)/(temp_prod-temp_froid)), 
                          name = "Volume équivalent")
    
    profil = pd.concat([profil, volume_eq], axis=1)

    return profil

tab_puisage = equivalence(tab_puisage, st.session_state.Tpuis, st.session_state.Tprod, st.session_state.Tfroid)

st.markdown('> **Données de puisage**')
gb = GridOptionsBuilder.from_dataframe(tab_puisage)
gb.configure_pagination()
gridOptions = gb.build()
AgGrid(tab_puisage, gridOptions=gridOptions, fit_columns_on_grid_load=True)

    
st.markdown("## Semi-instantané")
st.markdown("#### Synoptiques et typologie")

from PIL import Image

if st.checkbox('Montrer les synoptiques') :
    st.markdown("#### Sans bouclage")
    image = Image.open('C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/ECS_Semi_inst_Aucun.png')
    st.image(image)
    st.markdown("#### Raccord du bouclage entre le tiers supérieur et le milieu du ballon")
    image = Image.open('C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/ECS_Semi_inst_tiers_sup_milieu.png')
    st.image(image)
    st.markdown('#### Raccord du bouclage en bas du ballon')
    image = Image.open("C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/ECS_Semi_inst_Bas_stockage.png")
    st.image(image)
    st.markdown("#### Raccord du bouclage avant l'entrée échangeur")
    image = Image.open("C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/ECS_Semi_inst_av_entree_ech.png")
    st.image(image)
    st.markdown("#### Raccord du bouclage avant ballon")
    image = Image.open("C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/ECS_Semi_inst_avant_ball.png")
    st.image(image)
    
def Cp():
    " Capacité thermique de l'eau en Wh/(LK)"
    return 1.16

@st.cache
def energie_puisee(profil, temp_prod, temp_froid):
    "Fonction qui calcule l'énergie en kWh contenue dans l'eau à température de production"  
    energie_puisee = pd.Series(index=profil.index, 
                               data=profil['Volume équivalent']*Cp()*(temp_prod-temp_froid)/1000,
                               name = 'Energie puisée en kWh')

    profil = pd.concat([profil, energie_puisee], axis = 1)
    
    return profil

tab_puisage = energie_puisee(tab_puisage, st.session_state.Tprod, st.session_state.Tfroid)

##Méthode des besoins continus
st.markdown("#### Méthode des besoins continus")

@st.cache
def tableau_journalier(profil, freq) :
    ''' Fonction qui sépare les énergies et les volumes sur sept jours en 7 listes, une pour chaque jour '''
    nb = int(len(profil.index)/7) #Initialisation du nombre de données par jour
    
    df_0, df_1, df_2, df_3, df_4, df_5, df_6 = [], [], [], [], [], [], []
    vf_0, vf_1, vf_2, vf_3, vf_4, vf_5, vf_6 = [], [], [], [], [], [], []
    
    for k in profil.index :
        w = profil.loc[k]['Jour']
        if w == 0 :
            df_0.append(profil.loc[k]['Energie puisée en kWh'])
            vf_0.append(profil.loc[k]['Volume équivalent'])
        
        if w == 1 :
            df_1.append(profil.loc[k]['Energie puisée en kWh'])
            vf_1.append(profil.loc[k]['Volume équivalent'])
    
        if w == 2 :
            df_2.append(profil.loc[k]['Energie puisée en kWh'])
            vf_2.append(profil.loc[k]['Volume équivalent'])
           
        if w == 3 :
            df_3.append(profil.loc[k]['Energie puisée en kWh'])
            vf_3.append(profil.loc[k]['Volume équivalent'])
               
        if w == 4 :
            df_4.append(profil.loc[k]['Energie puisée en kWh'])
            vf_4.append(profil.loc[k]['Volume équivalent'])
                   
        if w == 5 :
            df_5.append(profil.loc[k]['Energie puisée en kWh'])
            vf_5.append(profil.loc[k]['Volume équivalent'])
                       
        if w == 6 :
            df_6.append(profil.loc[k]['Energie puisée en kWh'])
            vf_6.append(profil.loc[k]['Volume équivalent'])
           
    df_0 = pd.Series(data=df_0, index=[i for i in range(0,nb)], name='Lundi')
    df_1 = pd.Series(data=df_1, index=[i for i in range(0,nb)], name='Mardi')
    df_2 = pd.Series(data=df_2, index=[i for i in range(0,nb)], name='Mercredi')
    df_3 = pd.Series(data=df_3, index=[i for i in range(0,nb)], name='Jeudi')
    df_4 = pd.Series(data=df_4, index=[i for i in range(0,nb)], name='Vendredi')
    df_5 = pd.Series(data=df_5, index=[i for i in range(0,nb)], name='Samedi')
    df_6 = pd.Series(data=df_6, index=[i for i in range(0,nb)], name='Dimanche')
    
    df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6], axis=1)
    df['Heures'] = pd.Series(data=[i for i in np.arange(0,24, freq/60)])
    df.index = [k for k in range(0, len(df))]
    
    vf_0 = pd.Series(data=vf_0, index=[i for i in range(0,nb)], name='Lundi')
    vf_1 = pd.Series(data=vf_1, index=[i for i in range(0,nb)], name='Mardi')
    vf_2 = pd.Series(data=vf_2, index=[i for i in range(0,nb)], name='Mercredi')
    vf_3 = pd.Series(data=vf_3, index=[i for i in range(0,nb)], name='Jeudi')
    vf_4 = pd.Series(data=vf_4, index=[i for i in range(0,nb)], name='Vendredi')
    vf_5 = pd.Series(data=vf_5, index=[i for i in range(0,nb)], name='Samedi')
    vf_6 = pd.Series(data=vf_6, index=[i for i in range(0,nb)], name='Dimanche')
    
    vf = pd.concat([vf_0, vf_1, vf_2, vf_3, vf_4, vf_5, vf_6], axis=1)
    vf['Heures'] = pd.Series(data=[i for i in np.arange(0,24, freq/60)])
    vf.index = [k for k in range(0, len(df))]
    
    return df, vf

@st.cache
def besoins_consécutifs(profil, freq) :
    ''' Fonction qui prend en argument un tableau des énergies jour par jour, heure par heure, 
    correspondant au profil de puisage, et calcul les besoins consécutifs, une heure par une heure, 
    deux heures par deux heures, etc, avec début glissant (cf méthode)'''
    
    nb = int(len(profil.index)/7)
    
    tableau_energie, tableau_volumes = tableau_journalier(profil, freq)

    besoins = pd.DataFrame(data=0, index=[k for k in range(0,nb+1)], 
                           columns = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    
    volumes = pd.DataFrame(data=0, index=[k for k in range(0,nb+1)], 
                           columns = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    
    besoins.iloc[0]=0
    volumes.iloc[0]=0
    
    df_inter = pd.Series(data=0, index=[k for k in range(0,nb)])
    
    
    for j in tableau_energie.columns :
        k=1
        while k<=nb : 
            for i in tableau_energie.index :
                if (i+k)<=nb :
                    df_inter.iloc[i] = tableau_energie.loc[i:(i+k-1)][j].sum()
            besoins.loc[k][j] = df_inter.max()
            k=k+1
            df_inter=pd.Series(data=0, index=[k for k in range(0,nb)])        
    
    df_inter_vol = pd.Series(data=0, index=[k for k in range(0,nb)])
    
    for j in tableau_volumes.columns :
        k=1
        while k<=nb : 
            for i in tableau_volumes.index :
                if (i+k)<=nb :
                    df_inter_vol.iloc[i] = tableau_volumes.loc[i:(i+k-1)][j].sum()
            volumes.loc[k][j] = df_inter_vol.max()
            k=k+1
            df_inter_vol=pd.Series(data=0, index=[k for k in range(0,nb)])   
    
    return besoins, volumes

besoins, volumes = besoins_consécutifs(tab_puisage, int(st.session_state.freq))

def profil_consecutif(besoins, volumes, freq) :
    ''' Fonction qui prend en argument un tableau de besoins consécutifs pour chaque jour de la semaine,
    et renvoie un tableau prenant le max de chaque valeur '''

    besoins_max = pd.Series(data=besoins.max(axis=1), index=besoins.index, name = 'Max')
    volumes_max = pd.Series(data=volumes.max(axis=1), index=volumes.index, name = 'Max')
    
    lim = (freq/60)/10

    besoins_max = pd.concat([besoins_max, pd.Series(data=[i for i in np.arange(0,(24+lim), freq/60)], name='Heures')], axis=1)
    volumes_max = pd.concat([volumes_max, pd.Series(data=[i for i in np.arange(0,(24+lim), freq/60)], name='Heures')], axis=1)
        
    return besoins_max, volumes_max

besoins_max, volumes_max = profil_consecutif(besoins, volumes, int(st.session_state.freq))

def afficher_profil_consec(besoins, volumes):
    
    df_1 = pd.DataFrame({'Heures' : besoins['Heures'], 'Max' : besoins['Max'], 'Version' : 'Max énergies consécutives cumulées'})
    df_2 = pd.DataFrame({'Heures' : volumes['Heures'], 'Max' : volumes['Max'], 'Version' : 'Max volumes consécutifs cumulés'})

    df = pd.concat([df_1, df_2], axis=0)
      
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df_1['Heures'], y=df_1['Max'], name="Energies consécutives cumulées [Max]"),
        secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(x=df_2['Heures'], y=df_2['Max'], name="Volumes consécutifs cumulés [Max]"),
        secondary_y=True,
        )

    fig.update_layout(
        title_text="Energies et volumes maximum consécutifs cumulés max"
        )

    fig.update_xaxes(title_text="Heures")
    fig.update_xaxes(rangeslider_visible=True)

    fig.update_yaxes(title_text="Energie en kWh", secondary_y=False)
    fig.update_yaxes(title_text="Volume en L", secondary_y=True)

    #fig.show()
    
    return fig

fig3 = afficher_profil_consec(besoins_max, volumes_max)

@st.cache   
def liste_coeff_ech(besoins):
    '''Fonction qui prend en argument la liste des besoins max consécutifs et renvoie une droite de 
    puissance pour chaque point'''
    
    coeff_P_list = pd.Series(data=0.00, index=besoins.index, name = 'Liste des puissances échangeur')
    coeff_P_list[0]=0
    coeff_P_list.iloc[1:]=besoins['Max'].iloc[1:]/besoins['Heures'].iloc[1:]
    
    coeff_P_list.index = [k for k in range(0, len(coeff_P_list))]
    
    return coeff_P_list

coefficients_ech = liste_coeff_ech(besoins_max)

@st.cache
def tabl_diff(besoins):
    ''' Fonction qui prend en argument un coefficient de droite de puissance possible,
    qui calcule toutes les différences entre la courbe des puisages consécutifs et la droite de puissance
    de l'échangeur, pour chaque coefficient'''
    coeffs = liste_coeff_ech(besoins)

    tabl_diff = pd.DataFrame()
    tabl_diff.index = besoins.index

    tabl_diff['Heures']=besoins['Heures']

    count=0
    for j in coeffs.index :
        count=count+1
        tabl_diff["Coefficient"] = tabl_diff['Heures']*coeffs.iloc[j]
        tabl_diff["Coefficient"].index = besoins.index
        tabl_diff['Différence : '+str(count)]=besoins['Max']-tabl_diff["Coefficient"]
        del tabl_diff["Coefficient"]
        
    del tabl_diff['Heures']
    
    return tabl_diff

@st.cache
def max_diff(tabl_diff):
    ''' Fonction qui prend en argument un tableau de différences et cherche le maximum '''
    maxi = tabl_diff.max(axis=0)
    maxi.index= tabl_diff.index
    return maxi

@st.cache
def index_max_diff(tabl_diff, freq) :
    ''' Fonction qui prend en argument le tableau de différences et renvoie l'indice du maximum '''
    idmax = (tabl_diff.idxmax())*(freq/60)
    idmax.index = tabl_diff.index
    return idmax

@st.cache
def volume_stockage(maxi, dt, temp_prod, temp_froid, coeff) :
    ''' Fonction qui prend en argument une différence max et calcule le volume de stockage correspondant '''
    temp_prod = int(temp_prod)
    temp_froid = int(temp_froid)
    
    Vstock = 1000*maxi/(Cp()*(temp_prod-temp_froid)*coeff) 

    return Vstock

@st.cache
def liste_couplePV(liste_coeff, besoins, temp_prod, temp_froid, coeff, dt, freq, **kwargs):
    ''' Fonction qui prend en argument une liste de coeff de puissance échangeur et le tableau des besoins
    continus, et renvoie la liste des couples coeff,volume stockage'''
    couplesPV = pd.DataFrame(data=0.0, index=liste_coeff.index, columns = ['Puissance échangeur', 'Volume de stockage', 'Heure', 'Heure ballon vidé'])
    couplesPV['Heure'] = besoins['Heures']
    tabl = tabl_diff(besoins)
    Vstockage = volume_stockage(max_diff(tabl), dt, temp_prod, temp_froid, coeff)
    Vstockage.index = couplesPV.index
    couplesPV['Puissance échangeur'] = liste_coeff
    couplesPV['Volume de stockage'] = Vstockage
    empty = index_max_diff(tabl, freq)
    empty.index = couplesPV.index
    couplesPV['Heure ballon vidé'] = empty
    
    return couplesPV
    
couples = liste_couplePV(coefficients_ech, besoins_max, st.session_state.Tprod, st.session_state.Tfroid, st.session_state.coeff_stock, dt, int(st.session_state.freq))

def superposer_droites_et_puisage(liste_coeff, besoins):
    ''' Fonction qui prend en argument les coefficients directeurs des différentes droites de puissance
    échangeur et les besoins max, et renvoie le graphique des deux superposés'''
    
    droites_ech = pd.Series(data = besoins['Heures'], index=besoins.index, name = 'Heures')
    Heures = pd.Series(data = besoins['Heures'], index=besoins.index, name='Heures')
    
    for k in liste_coeff.index :
        P = pd.Series(data=besoins['Heures']*liste_coeff[k], index=besoins.index, name = 'Pech avec coeff '+str(liste_coeff[k]))
        droites_ech = pd.concat([droites_ech, P], axis=1)

    droites_ech['Version']='Puissance échangeur'
    besoins['Version']='Puisages consécutifs'
    

    fig = px.line(besoins, x='Heures', y='Max', 
                  labels={'Heures' : 'Heures', 
                          'Max' : 'Energie puisée en kWh'})
    
    for j in droites_ech.columns[1:] :
        fig.add_scatter(x=droites_ech['Heures'], y=droites_ech[j], mode='lines')
        
    fig.update_xaxes(rangeslider_visible=True)
        
    #fig.show()
    
    return fig

fig4 = superposer_droites_et_puisage(coefficients_ech, besoins_max)

def courbe_egal_satis_besoins(couplesPV):
    fig = px.scatter(couplesPV, x='Volume de stockage', y='Puissance échangeur', color='Heure',
                  labels={ 'Puissance échangeur' : "Puissance de l'échangeur en kW",
                          'Volume de stockage' : 'Volume de stockage en litres'},
                  title = "Courbe d'égale satisfaction des besoins", color_continuous_scale=px.colors.sequential.Blugrn)
    
    fig.update_xaxes(rangeslider_visible=True)
    
    #fig.show()
    
    return fig

fig5 = courbe_egal_satis_besoins(couples)

st.markdown("> **Courbe d'égale satisfaction des besoins**")
st.plotly_chart(fig5, use_container_width=True)

gb3 = GridOptionsBuilder.from_dataframe(couples)
gb3.configure_pagination()
gridOptions3 = gb3.build()
res1 = AgGrid(couples, gridOptions=gridOptions3, fit_columns_on_grid_load=True)

if st.checkbox("Afficher les résultats détaillés de la méthode des besoins continus") :
    st.subheader("Besoins consécutifs")
    st.plotly_chart(fig3, use_container_width=True)
    st.subheader('Droites de puisage')
    st.plotly_chart(fig4, use_container_width=True)

##Méthode ASHRAE
st.markdown("#### Méthodes ASHRAE")

@st.cache
def besoins_consécutifs_simples(profil, freq) : 
    ''' Fonction qui prend en argument un profil, et calcule la somme des besoins consécutifs SANS faire glisser
    le début de puisage '''
    nb = int(len(profil.index)/7) #Initialisation du nombre de données par jour
    
    besoins = pd.DataFrame(data=0, index=[k for k in range(0,nb+1)], 
                           columns = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    
    volumes = pd.DataFrame(data=0, index=[k for k in range(0,nb+1)], 
                           columns = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    

    df_0, df_1, df_2, df_3, df_4, df_5, df_6 = [], [], [], [], [], [], []
    vf_0, vf_1, vf_2, vf_3, vf_4, vf_5, vf_6 = [], [], [], [], [], [], []
    
    for k in profil.index :
        w = profil.loc[k]['Jour']
        if w == 0 :
            df_0.append(profil.loc[k]['Energie puisée en kWh'])
            vf_0.append(profil.loc[k]['Volume équivalent'])
        
        if w == 1 :
            df_1.append(profil.loc[k]['Energie puisée en kWh'])
            vf_1.append(profil.loc[k]['Volume équivalent'])
    
        if w == 2 :
            df_2.append(profil.loc[k]['Energie puisée en kWh'])
            vf_2.append(profil.loc[k]['Volume équivalent'])
           
        if w == 3 :
            df_3.append(profil.loc[k]['Energie puisée en kWh'])
            vf_3.append(profil.loc[k]['Volume équivalent'])
               
        if w == 4 :
            df_4.append(profil.loc[k]['Energie puisée en kWh'])
            vf_4.append(profil.loc[k]['Volume équivalent'])
                   
        if w == 5 :
            df_5.append(profil.loc[k]['Energie puisée en kWh'])
            vf_5.append(profil.loc[k]['Volume équivalent'])
                       
        if w == 6 :
            df_6.append(profil.loc[k]['Energie puisée en kWh'])
            vf_6.append(profil.loc[k]['Volume équivalent'])
           
    df_0 = pd.Series(data=df_0, index=[i for i in range(0,nb)], name='Lundi')
    df_1 = pd.Series(data=df_1, index=[i for i in range(0,nb)], name='Mardi')
    df_2 = pd.Series(data=df_2, index=[i for i in range(0,nb)], name='Mercredi')
    df_3 = pd.Series(data=df_3, index=[i for i in range(0,nb)], name='Jeudi')
    df_4 = pd.Series(data=df_4, index=[i for i in range(0,nb)], name='Vendredi')
    df_5 = pd.Series(data=df_5, index=[i for i in range(0,nb)], name='Samedi')
    df_6 = pd.Series(data=df_6, index=[i for i in range(0,nb)], name='Dimanche')
    
    df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6], axis=1)
    
    vf_0 = pd.Series(data=vf_0, index=[i for i in range(0,nb)], name='Lundi')
    vf_1 = pd.Series(data=vf_1, index=[i for i in range(0,nb)], name='Mardi')
    vf_2 = pd.Series(data=vf_2, index=[i for i in range(0,nb)], name='Mercredi')
    vf_3 = pd.Series(data=vf_3, index=[i for i in range(0,nb)], name='Jeudi')
    vf_4 = pd.Series(data=vf_4, index=[i for i in range(0,nb)], name='Vendredi')
    vf_5 = pd.Series(data=vf_5, index=[i for i in range(0,nb)], name='Samedi')
    vf_6 = pd.Series(data=vf_6, index=[i for i in range(0,nb)], name='Dimanche')
    
    vf = pd.concat([vf_0, vf_1, vf_2, vf_3, vf_4, vf_5, vf_6], axis=1)
    
    besoins.iloc[0]=0
    besoins.iloc[1]=df.iloc[0]

    for k in range(1, len(besoins.index)-1):
        besoins.loc[k+1]['Lundi']=besoins.loc[k]['Lundi']+df.loc[k]['Lundi']
        besoins.loc[k+1]['Mardi']=besoins.loc[k]['Mardi']+df.loc[k]['Mardi']
        besoins.loc[k+1]['Mercredi']=besoins.loc[k]['Mercredi']+df.loc[k]['Mercredi']
        besoins.loc[k+1]['Jeudi']=besoins.loc[k]['Jeudi']+df.loc[k]['Jeudi']
        besoins.loc[k+1]['Vendredi']=besoins.loc[k]['Vendredi']+df.loc[k]['Vendredi']
        besoins.loc[k+1]['Samedi']=besoins.loc[k]['Samedi']+df.loc[k]['Samedi']
        besoins.loc[k+1]['Dimanche']=besoins.loc[k]['Dimanche']+df.loc[k]['Dimanche']
        
    volumes.iloc[0]=0
    volumes.iloc[1]=vf.iloc[0]

    for k in range(1, len(volumes.index)-1):
        volumes.loc[k+1]['Lundi']=volumes.loc[k]['Lundi']+vf.loc[k]['Lundi']
        volumes.loc[k+1]['Mardi']=volumes.loc[k]['Mardi']+vf.loc[k]['Mardi']
        volumes.loc[k+1]['Mercredi']=volumes.loc[k]['Mercredi']+vf.loc[k]['Mercredi']
        volumes.loc[k+1]['Jeudi']=volumes.loc[k]['Jeudi']+vf.loc[k]['Jeudi']
        volumes.loc[k+1]['Vendredi']=volumes.loc[k]['Vendredi']+vf.loc[k]['Vendredi']
        volumes.loc[k+1]['Samedi']=volumes.loc[k]['Samedi']+vf.loc[k]['Samedi']
        volumes.loc[k+1]['Dimanche']=volumes.loc[k]['Dimanche']+vf.loc[k]['Dimanche']
    
    return besoins, volumes

besoins2, volumes2 = besoins_consécutifs_simples(tab_puisage, int(st.session_state.freq))

besoins_max2, volumes_max2 = profil_consecutif(besoins2, volumes2, int(st.session_state.freq))

@st.cache
def afficher_profil_consec_simple(besoins, volumes):
    
    df_1 = pd.DataFrame({'Heures' : besoins['Heures'], 'Max' : besoins['Max'], 'Version' : 'Energies consécutives cumulées'})
    df_2 = pd.DataFrame({'Heures' : volumes['Heures'], 'Max' : volumes['Max'], 'Version' : 'Volumes consécutifs cumulés'})

    df = pd.concat([df_1, df_2], axis=0)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df_1['Heures'], y=df_1['Max'], name="Energies consécutives cumulées"),
        secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(x=df_2['Heures'], y=df_2['Max'], name="Volumes consécutifs cumulés"),
        secondary_y=True,
        )

    fig.update_layout(
        title_text="Energies et volumes maximum consécutifs cumulés"
        )

    fig.update_xaxes(title_text="Heures")
    fig.update_xaxes(rangeslider_visible=True)

    fig.update_yaxes(title_text="Energie en kWh", secondary_y=False)
    fig.update_yaxes(title_text="Volume en L", secondary_y=True)

    #fig.show()
    
    return fig

fig6 = afficher_profil_consec_simple(besoins_max2, volumes_max2)

@st.cache
def methode_ASHRAE(profil, temp_froid, temp_prod, besoins_max, volumes_max, coeff) :
    ''' Fonction qui calcule les couples (P,V) à partir de la méthode 2019 ASHRAE Handbook -
    HVAC Applications, chapter 51, page 51.16 '''
    temp_prod = int(temp_prod)
    temp_froid = int(temp_froid)
    
    
    CouplesPV = pd.DataFrame(data=0.0, index=volumes_max.index, columns = ['Heure','Puissance échangeur', 'Volume de stockage'])

    CouplesPV['Heure'] = volumes_max['Heures']

    for k in range(0, len(volumes_max.index)-1) :
        CouplesPV.at[k,'Volume de stockage'] = volumes_max['Max'].iloc[k]/coeff
        CouplesPV.at[k,'Puissance échangeur'] = (volumes_max['Max'].iloc[k+1]-volumes_max['Max'].iloc[k])*Cp()*(temp_prod-temp_froid)/(1000*(CouplesPV.at[(k+1),'Heure']-CouplesPV.at[k,'Heure']))
   
    CouplesPV['Puissance échangeur'].iat[-1] = 0
    CouplesPV['Volume de stockage'].iat[-1] = volumes_max['Max'].iloc[-1]/coeff
    
    average_24 = volumes_max['Max'].iat[-1]*Cp()*(temp_prod-temp_froid)/(CouplesPV.iat[-1,0]*1000)
    
    CouplesPV['Puissance échangeur'] = CouplesPV['Puissance échangeur'].mask(CouplesPV['Puissance échangeur']<average_24, average_24)
    
    return CouplesPV

@st.cache
def methode_ASHRAE_simple(profil, temp_froid, temp_prod, besoins_max, volumes_max, coeff):
    temp_prod = int(temp_prod)
    temp_froid = int(temp_froid)
    
    CouplesPV = pd.DataFrame(data=0.0, index=volumes_max.index, columns = ['Heure', 'Puissance échangeur', 'Volume de stockage'])
    
    CouplesPV['Heure'] = volumes_max['Heures']
    
    for k in CouplesPV.index:
        if k!= 0 :
            CouplesPV.at[k,'Puissance échangeur'] = volumes_max['Max'].iloc[k]*Cp()*(temp_prod-temp_froid)/(1000*CouplesPV.at[k,'Heure'])
            CouplesPV.at[k,'Volume de stockage'] = volumes_max['Max'].iloc[k]/coeff
     
    return CouplesPV

def verif_couples_ASHRAE(Couples) :
    ''' Fonction qui prend en argument des couples issus du calcul ASHRAE et ne conserve que ceux qui 
    fonctionnent '''
    Couples = Couples.sort_values(by=['Volume de stockage'], ascending=False)
    
    new_couples = Couples

    for k in range(len(Couples.index)-1, 0, -1) :
        if new_couples.loc[k-1]['Puissance échangeur'] < new_couples.loc[k]['Puissance échangeur'] :
            new_couples = new_couples.drop(k-1)
            new_couples.index = [k for k in range(len(new_couples.index)-1, -1, -1)]
    
    return new_couples

couples_ASHRAE = methode_ASHRAE_simple(tab_puisage, st.session_state.Tfroid, st.session_state.Tprod, besoins_max2, volumes_max2, st.session_state.coeff_stock)
couples_ASHRAE2 = methode_ASHRAE(tab_puisage, st.session_state.Tfroid, st.session_state.Tprod, besoins_max2, volumes_max2, st.session_state.coeff_stock)

couples_ASHRAE = verif_couples_ASHRAE(couples_ASHRAE)
couples_ASHRAE2 = verif_couples_ASHRAE(couples_ASHRAE2)

fig7 = courbe_egal_satis_besoins(couples_ASHRAE)

fig8 = courbe_egal_satis_besoins(couples_ASHRAE2)

couples_ASHRAE["Puissance échangeur"] = couples_ASHRAE["Puissance échangeur"].astype(str)
couples_ASHRAE["Volume de stockage"] = couples_ASHRAE["Volume de stockage"].astype(str)
couples_ASHRAE["Heure"] = couples_ASHRAE["Heure"].astype(str)

couples_ASHRAE2["Puissance échangeur"] = couples_ASHRAE2["Puissance échangeur"].astype(str)
couples_ASHRAE2["Volume de stockage"] = couples_ASHRAE2["Volume de stockage"].astype(str)
couples_ASHRAE2["Heure"] = couples_ASHRAE2["Heure"].astype(str)

st.markdown("> **Courbe d'égale satisfaction des besoins : ASHRAE simple**")
st.plotly_chart(fig7, use_container_width=True)
gb = GridOptionsBuilder.from_dataframe(couples_ASHRAE)
gb.configure_pagination()
gridOptions = gb.build()
res2 = AgGrid(couples_ASHRAE, gridOptions=gridOptions, fit_columns_on_grid_load=True)
st.markdown("> **Courbe d'égale satisfaction des besoins : ASHRAE précise**")
st.plotly_chart(fig8, use_container_width=True)
gb = GridOptionsBuilder.from_dataframe(couples_ASHRAE2)
gb.configure_pagination()
gridOptions = gb.build()
res3 = AgGrid(couples_ASHRAE2, gridOptions=gridOptions, fit_columns_on_grid_load=True)


if st.checkbox("Afficher les résultats détaillés des méthodes ASHRAE") :
    st.markdown("> **Besoins consécutifs**")
    st.plotly_chart(fig6, use_container_width=True)

##Méthode des besoins discontinus
st.markdown("#### Méthode des besoins discontinus")

st.sidebar.header("Semi-instantané : méthode des besoins discontinus")
st.sidebar.number_input("Entrez la fraction de consommation journalière pendant la pointe de puisage", min_value = 0.01, max_value=1.0, value=0.65, step=0.01, key='fraction')


def conso_journaliere_max(profil) :
    ''' Fonction qui prend un profil hebdomaire, et renvoie la conso journaliere max sur la semaine '''
    new_profil = profil.copy()
    new_profil['Date2'] = 0
    for k in profil.index :
        d = new_profil['Date'].iloc[k].date().__str__()
        new_profil['Date2'].iloc[k] = d
        
    d = new_profil['Date2'].iloc[0]
    Vmax = (new_profil[new_profil['Date2'] == str(d)]['Volume équivalent']).sum()
    
    for j in range(1, len(profil.index)):
        if new_profil['Date2'].iloc[j] != str(d) :
            d = new_profil['Date2'].iloc[j]
            V = (new_profil[new_profil['Date2'] == str(d)]['Volume équivalent']).sum()
            Vmax = max(Vmax, V)
    
    return Vmax

@st.cache
def pointe(profil, fraction) :
    fraction = float(fraction)
    Vmax = conso_journaliere_max(profil)
    Vmax = fraction*Vmax
    return Vmax

conso_pointe = pointe(tab_puisage, st.session_state.fraction)

st.sidebar.number_input("Entrez la durée de la pointe de puisage", min_value=0.1, value=2.0, step=0.1, key="duree_pointe")
st.sidebar.number_input("Entrez la durée de reconstitution du stockage", min_value=0.1, max_value=20.0, value=8.0, step=0.1, key="duree_rec")
st.sidebar.number_input("Entrez le volume de stockage", min_value=1, value = 1000, key="Stockage_disc")

@st.cache
def besoins_discontinus(vol_pointe, duree_pointe, duree_rec, stockage, coeff, Tstock, Tprod, Tfroid, profil) :
    ''' Fonction qui prend en argument un volume de pointe, une durée de puisage de pointe, et une durée
    de reconstitution d'un certain volume de stockage, et renvoie la puissance échangeur nécessaire en 
    considérant que tout le volume de stockage est consommé durant la pointe'''
    #vol_pointe, stockage en L
    #duree_pointe, duree_rec en heure
    #Tstock, Tprod et Tfroid en °C
    
    stock_red = stockage*coeff
    #Vidage du stock
    Pech1 = (Cp()*10**(-3)*vol_pointe*(Tprod-Tfroid)-Cp()*10**(-3)*stock_red*(Tstock-Tfroid))/duree_pointe
    Pech = Pech1

    #Reconstitution du stock (Attention, hypothèse : puisages discontinus, donc on suppose que l'échangeur
    #ne sert qu'à reconstituer le stock)
    Pech2 = Cp()*10**(-3)*stock_red*(Tprod-Tfroid)/duree_rec

    #Puissance échangeur mini nécessaire (=puissance sur 20h)
    Pech3 = Cp()*10**(-3)*(Tprod-Tfroid)*conso_journaliere_max(profil)/20

    #On prend le max des deux
    if Pech2 > Pech :
        Pech = Pech2
    if Pech3 > Pech :
        Pech= Pech3
    #Pech = Pech.mask(Pech2>Pech, Pech2)
    #Pech = Pech.mask(Pech3>Pech, Pech3)

    Couples_disc = pd.concat([pd.Series(data=Pech, index=[0]), pd.Series(data=stockage, index=[0])], axis=1)
    Couples_disc.name = 'Couples (P,V) - Puisages discontinus'
    Couples_disc.columns = ['Puissance échangeur', 'Volume de stockage']
    
    return Couples_disc

Couples_disc = besoins_discontinus(conso_pointe, st.session_state.duree_pointe, st.session_state.duree_rec, st.session_state.Stockage_disc, st.session_state.coeff_stock, st.session_state.Tstock, st.session_state.Tprod, st.session_state.Tfroid, tab_puisage)

@st.cache
def afficher_PV(Couples_disc) :
    fig = px.scatter(Couples_disc, x="Volume de stockage", y="Puissance échangeur", color=Couples_disc.index,
                     labels = { 'Puissance échangeur' : 'Puissance échangeur en kW',
                               'Volume de stockage' : 'Volume de stockage en L'},
                     title = 'Couples (P,V) - Puisages discontinus', color_continuous_scale=px.colors.sequential.Blugrn)
    fig.update_xaxes(rangeslider_visible=True)
    
    #fig.show()
    
    return fig

fig9 = afficher_PV(Couples_disc)
gb = GridOptionsBuilder.from_dataframe(Couples_disc)
gb.configure_pagination()
gridOptions = gb.build()
res6 = AgGrid(Couples_disc, gridOptions=gridOptions, fit_columns_on_grid_load=True)

if st.checkbox("Afficher les résultats détaillés de la méthode des besoins discontinus") :
    st.markdown("> **Puissance échangeur et volume de stockage**")
    st.plotly_chart(fig9, use_container_width=True)

   
##Accumulation pure
st.markdown("## Accumulation pure")
st.sidebar.header("Accumulation pure")
Vmax = conso_journaliere_max(tab_puisage)
st.sidebar.number_input("Entrez la durée de reconstitution du stockage", min_value=0.1, max_value=20.0, value=8.0, step=0.01, key="temps_reconstitution")

@st.cache
def Accumulation(profil, Vmax, coeff, temp_prod, temp_froid, temp_stock, temps_rec) :    
    #Calcul de l'énergie nécessaire
    Emax = Cp()*Vmax*(temp_prod-temp_froid)*10**(-3) #en kWh
    
    #Calcul du volume correspondant
    V = 10**3*Emax/(Cp()*(temp_stock-temp_froid)*coeff) #en Litres
    #print("Le volume de stockage nécessaire est de "+str(V)+" L.")
    
    #Calcul de la puissance échangeur pour remplir le ballon sur la durée rec sans tenir compte des pertes du ballons et du réseau
    P=Emax/temps_rec
    #print("La puissance échangeur nécessaire pour réchauffer le stockage en "+str(temps_rec)+" heures est de "+str(P)+" kW.")
    
    df = pd.DataFrame({'Volume de stockage' : [V] , 'Puissance échangeur' : [P], 'Puisage max journalier' : [Vmax], 'Temps de reconstitution du stock' : [temps_rec]})
    
    return df

df = Accumulation(tab_puisage, Vmax, st.session_state.coeff_stock, st.session_state.Tprod, st.session_state.Tfroid, st.session_state.Tstock, st.session_state.temps_reconstitution)
  
st.markdown("> **Volume et puissance de réchauffage**")
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination()
gridOptions = gb.build()
res4 = AgGrid(df, gridOptions=gridOptions, fit_columns_on_grid_load=True)
    
##Instantané
st.markdown("## Instantané")

@st.cache
def conso_freq_max(profil) :
    ''' Fonction qui prend un argument en profil et renvoie le pic horaire le plus important sur la semaine'''
    Pic = profil['Volume équivalent'].max()
    return Pic

pointe_inst = conso_freq_max(tab_puisage)

@st.cache
def instantané(pointe, temp_prod, temp_froid, profil, freq) :
    temp_prod = int(temp_prod)
    temp_froid = int(temp_froid)
    
    duree = freq/60
    
    Pech = pointe*Cp()*10**(-3)*(temp_prod-temp_froid)/duree
    #print("La puissance échangeur nécessaire est "+str(Pech)+" kW.")
    
    return Pech

Pinst = instantané(pointe_inst, st.session_state.Tprod, st.session_state.Tfroid, tab_puisage, st.session_state.freq)

DFinst = pd.DataFrame(data=[[Pinst, 0]], index=[0], columns=["Puissance échangeur", "Volume de stockage"])


st.markdown("> **Puissance instantanée**")
gb = GridOptionsBuilder.from_dataframe(DFinst)
gb.configure_pagination()
gridOptions = gb.build()
res5 = AgGrid(DFinst, gridOptions=gridOptions, fit_columns_on_grid_load=True)

st.markdown("## Choix des paramètres")
option = st.selectbox(
    'Méthode', ('Méthode des besoins continus', 'Méthode ASHRAE - simple', 'Méthode ASHRAE - précise', 'Méthode des besoins discontinus', 'Accumulation pure', 'Instantané'))


if option == 'Méthode des besoins continus' :
    AgGrid(res1['data'], fit_columns_on_grid_load=True)
elif option == 'Méthode ASHRAE - simple' :
    AgGrid(res2['data'], fit_columns_on_grid_load=True)
elif option == 'Méthode ASHRAE - précise' :
    AgGrid(res3['data'], fit_columns_on_grid_load=True)
elif option == 'Méthode des besoins discontinus' :
    AgGrid(res6['data'], fit_columns_on_grid_load=True)
elif option == 'Accumulation pure' :
    AgGrid(res4['data'], fit_columns_on_grid_load=True)
elif option == 'Instantané' :
    AgGrid(res5['data'], fit_columns_on_grid_load=True)

col1, col2 = st.columns(2)

with col1 :
    st.number_input("Puissance de l'échangeur", min_value=0.0, step=0.01, key="Pech")
with col2 :
    st.number_input("Volume de stockage", min_value = 0.0, step=0.1, key="Vstock")


## Géométrie
st.markdown("## Géométrie")
st.markdown("> **Création du bouclage**")
st.sidebar.header("Géométrie - Bouclage")
st.sidebar.number_input("Température départ de bouclage", min_value=50.0, value=60.0, step=0.01, key="Tfluid_aller")
st.sidebar.number_input("Température de retour de bouclage", min_value=50.0, value=55.0, step=0.01, key="Tfluid_retour")
st.sidebar.number_input("Diamètre intérieur du réseau de charge (mm) - système SI", min_value=0.01, value=65.0, step=0.01, key="Dint_charge")

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns = ["Longueur aller (m)", "Diamètre intérieur aller (mm)", "Longueur retour (m)", "Diamètre intérieur retour (mm)"])
    
ncol = st.session_state.df.shape[1]  # col count

with st.form(key="add form", clear_on_submit= True):
    st.subheader("Ajouter un bouclage")
    cols = st.columns(ncol)
    rwdta = []

    for i in range(ncol):
        rwdta.append(cols[i].number_input(f"{st.session_state.df.columns[i]}"))

    if st.form_submit_button("Ajouter"):
        rw = st.session_state.df.shape[0] + 1
        st.session_state.df.loc[rw] = rwdta
        st.success("Bouclage ajouté")
    
    if st.form_submit_button("Effacer le dernier bouclage") :
        st.session_state.df.drop(st.session_state.df.index[-1], inplace=True)
        st.warning("Le dernier bouclage a bien été effacé")

st.dataframe(st.session_state.df)

st.markdown("## Isolation réseaux")
st.markdown("> **Tableaux des coefficients de pertes thermiques linéiques (fonction du type de matériau, de l'isolant et de son épaisseur**")

if st.checkbox("Afficher les tableaux d'isolation") :
    image = Image.open("C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/Tubes_Cuivre.png")
    st.image(image)
    image = Image.open("C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/Tubes_Cuivre_calorifuge.png")
    st.image(image)
    image = Image.open("C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/Tubes_multicouches.png") 
    st.image(image)
    image= Image.open("C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/Tubes_multicouches_calorifuge.png")
    st.image(image)
    image = Image.open("C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/Tubes_PVC_C.png")
    st.image(image)
    image = Image.open("C:/Users/RomaneTEZE/Dropbox (LHIRR)/Base/RTE-Stage calcul/Code stage/Outil ECS/Images/Tubes_PVC_C_calorifuge.png")
    st.image(image)
    
#if "df2" not in st.session_state:
 #   st.session_state.df2 = pd.DataFrame(columns = ["Coefficient de pertes thermiques linéiques aller (W/(mK)", "Coefficient de pertes thermiques linéiques retour (W/(mK)"])

df2 = pd.DataFrame(data=0, index=st.session_state.df.index, columns = ["Numéro de bouclage", "Coefficient de pertes thermiques linéique aller (W/(mK))", "Coefficient de pertes thermiques linéique retour (W/(mK))"])
df2['Numéro de bouclage'] = st.session_state.df.index
gb = GridOptionsBuilder.from_dataframe(df2)
gb.configure_column('Coefficient de pertes thermiques linéique aller (W/(mK))', editable=True)
gb.configure_column('Coefficient de pertes thermiques linéique retour (W/(mK))', editable=True)
grid_options = gb.build()
grid_response = AgGrid(df2, gridOptions=grid_options, fit_columns_on_grid_load=True, update_mode = 'MODEL_CHANGED', data_return_mode='AS_INPUT')

st.session_state.df2_filled = grid_response['data']
st.session_state.df2_filled.index = [u for u in range(1, len(st.session_state.df2_filled.index)+1)]
    
#ncol2 = st.session_state.df2.shape[1]  # col count

#with st.form(key="add form 2", clear_on_submit= True):
 #   st.subheader("Ajouter les coefficients des isolants de bouclage")
  #  st.info("Ajoutez les coefficients aller et retour pour chaque réseau de bouclage précédemment défini")
   # cols2 = st.columns(ncol2)
    #rwdta2 = []

    #for i in range(ncol2):
     #   rwdta2.append(cols2[i].number_input(f"{st.session_state.df2.columns[i]}"))

    #if st.form_submit_button("Ajouter"):
     #   rw2 = st.session_state.df2.shape[0] + 1
      #  st.session_state.df2.loc[rw2] = rwdta2
       # st.success("Caractéristiques ajoutées")
    
    #if st.form_submit_button("Effacer la dernière ligne") :
     #   st.session_state.df2.drop(st.session_state.df2.index[-1], inplace=True)
      #  st.warning("La dernière ligne a bien été effacé")
        
#st.dataframe(st.session_state.df2)

st.markdown("## Isolation ballons")

if "df3" not in st.session_state:
    st.session_state.df3 = pd.DataFrame(columns = ["Volume du ballon (L)", "Hauteur du ballon (m)", "Coefficient de pertes thermiques par mètre d'isolant (W/(mK))", "Epaisseur de l'isolant (en m)"])
    
ncol3 = st.session_state.df3.shape[1]  # col count

with st.form(key="add form 3", clear_on_submit= True):
    st.subheader("Ajouter les caractéristiques de l'isolation ballon")
    st.info("Ajoutez autant de caractéristiques que de ballon dans le système")
    cols3 = st.columns(ncol3)
    rwdta3 = []

    for i in range(ncol3):
        rwdta3.append(cols3[i].number_input(f"{st.session_state.df3.columns[i]}"))

    if st.form_submit_button("Ajouter"):
        rw3 = st.session_state.df3.shape[0] + 1
        st.session_state.df3.loc[rw3] = rwdta3
        st.success("Caractéristiques ajoutées")
    
    if st.form_submit_button("Effacer la dernière ligne") :
        st.session_state.df3.drop(st.session_state.df3.index[-1], inplace=True)
        st.warning("La dernière ligne a bien été effacé")
        
st.dataframe(st.session_state.df3)

if st.session_state.df3["Volume du ballon (L)"].sum() > st.session_state.Vstock :
    st.error("Attention ! La somme des volumes des ballons est plus grande que le volume de stockage !")
if st.session_state.df3["Volume du ballon (L)"].sum() < st.session_state.Vstock :
    st.warning("Attention ! La somme des volumes des ballons est plus faible que le volume de stockage !")

def déperditions(temp_fluid1, isol1, L1, temp_fluid2, isol2, L2, tamb) :
    ''' Calcule les déperditions thermiques dans le circuit de bouclage (aller et retour) '''
    deperditions1 = (temp_fluid1 - tamb)*isol1*L1
    deperditions2 = (temp_fluid2 - tamb)*isol2*L2
    
    deperditions = (deperditions1+deperditions2)*10**(-3)
    
    return deperditions

st.markdown("## Calcul des pertes de bouclage")


st.session_state.df4 = pd.DataFrame(index=st.session_state.df.index, columns = ["Numéro de bouclage", "Pertes thermiques en kW"])

nbouclage = st.session_state.df.shape[0]

for i in range(1, nbouclage+1) :
    st.session_state.df4.loc[i,"Numéro de bouclage"] = i
    st.session_state.df4.loc[i,"Pertes thermiques en kW"] = déperditions(st.session_state.Tfluid_aller, st.session_state.df2_filled.loc[i]["Coefficient de pertes thermiques linéique aller (W/(mK))"], st.session_state.df.loc[i]["Longueur aller (m)"], st.session_state.Tfluid_retour, st.session_state.df2_filled.loc[i]["Coefficient de pertes thermiques linéique retour (W/(mK))"], st.session_state.df.loc[i]["Longueur retour (m)"], st.session_state.Tamb)
    
st.dataframe(st.session_state.df4)

pertes_tot = st.session_state.df4["Pertes thermiques en kW"].sum()

st.markdown("> **Les pertes de bouclage totales sont de** "+str(pertes_tot)+" **kW**")

st.markdown("## Calcul des pertes de stockage")

def pertes_bal(tamb, tstock, isol, epaisseur, vol, hauteur) :
    Dint = sqrt(4*vol*10**(-3)/(hauteur*pi))
    Se = pi*Dint*hauteur + 2*pi*Dint**2/4
    
    pertes_kW = (tstock-tamb)*(isol/epaisseur)*Se*10**(-3)
    
    return pertes_kW


st.session_state.df5 = pd.DataFrame(index = st.session_state.df3.index, columns = ["Numéro de ballon", "Pertes thermiques en kW"])

nballon = st.session_state.df3.shape[0]

for i in range(1, nballon+1) :
    st.session_state.df5.loc[i,"Numéro de ballon"] = i
    st.session_state.df5.loc[i,"Pertes thermiques en kW"] = pertes_bal(st.session_state.Tamb, st.session_state.Tstock, st.session_state.df3.loc[i]["Coefficient de pertes thermiques par mètre d'isolant (W/(mK))"], st.session_state.df3.loc[i]["Epaisseur de l'isolant (en m)"], st.session_state.df3.loc[i]["Volume du ballon (L)"], st.session_state.df3.loc[i]["Hauteur du ballon (m)"])
    
st.dataframe(st.session_state.df5)

pertes_tot_ball = st.session_state.df5["Pertes thermiques en kW"].sum()

st.markdown("> **Les pertes de stockage totales sont de** "+str(pertes_tot_ball)+" **kW**")
                                
def debit_boucl(deperditions, temp_fluid1, temp_fluid2, Dint) :
    ''' Calcule le débit de bouclage en fonction des déperditions thermiques du circuit de bouclage '''
    #en L/h
    debit = deperditions*3600/((temp_fluid1-temp_fluid2)*4.185)
    
    vitesse = debit*10**(-3)/(3600*pi*(Dint*10**(-3))**2/4)
    dv = pd.DataFrame(data=0, index=[0], columns = ["Débit de bouclage en L/h", "Vitesse dans le bouclage en m/s"])
    dv['Débit de bouclage en L/h'] = debit
    dv["Vitesse dans le bouclage en m/s"] = vitesse
    
    return dv


st.session_state.df6 = pd.DataFrame(index = st.session_state.df.index, columns = ["Numéro de bouclage", "Débit de bouclage (L/h)", "Vitesse dans le bouclage (m/s)"])
    
nbouclage = st.session_state.df.shape[0]

for i in range(1, nbouclage+1) :
    st.session_state.df6.loc[i,'Numéro de bouclage'] = i
    st.session_state.df6.loc[i,"Débit de bouclage (L/h)"] = debit_boucl(st.session_state.df4.loc[i]["Pertes thermiques en kW"], st.session_state.Tfluid_aller, st.session_state.Tfluid_retour, st.session_state.df.loc[i]["Diamètre intérieur retour (mm)"]).loc[0,'Débit de bouclage en L/h']
    st.session_state.df6.loc[i,"Vitesse dans le bouclage (m/s)"] = debit_boucl(st.session_state.df4.loc[i]["Pertes thermiques en kW"], st.session_state.Tfluid_aller, st.session_state.Tfluid_retour, st.session_state.df.loc[i]["Diamètre intérieur retour (mm)"]).loc[0,'Vitesse dans le bouclage en m/s']
    
st.dataframe(st.session_state.df6)

errors = 0
for k in st.session_state.df6.index :
    if (st.session_state.df6.loc[k]["Vitesse dans le bouclage (m/s)"]<0.2)|(st.session_state.df6.loc[k]["Vitesse dans le bouclage (m/s)"]>0.5) :
        st.error("La vitesse dans le bouclage "+str(k)+" n'est pas comprise entre 0,2 et 0,5 m/s. Merci de modifier les paramètres")
        errors =+ 1

Qbouclage = st.session_state.df6["Débit de bouclage (L/h)"].sum()
        
if (errors == 0)&(Qbouclage!=0) :
    st.success("Les vitesses de tous les bouclages sont comprises entre 0,2 et 0,5 m/s !")

st.markdown("> **Le débit de bouclage total nécessaire est de** "+str(Qbouclage)+" **L/h**")

st.sidebar.header("Caractéristiques de l'installation")
st.sidebar.checkbox(label = "Réchauffage de bouclage indépendant", value=False, key="réchauffage_indépendant")
st.sidebar.selectbox("Système", ("SI", "I", "ACC"), key='type_système')
st.sidebar.selectbox("Type de bouclage", ('avant_entree_ech', 'bas_stockage', 'tiers_sup_milieu', 'avant_ballon', 'Aucun'), key='type_bouclage')

def Calcul_Pech_Qcharge(système, type_bouclage, réchauffage, Pech, pertes_boucl, pertes_bal, qboucle, tfroid, tprod, Dint) :
    ''' Calcule le débit de charge et met à jour la puissance échangeur pour contrer les pertes de ballons
    et les pertes de bouclage '''
    ''' Dépend du type de bouclage '''
    dis = pd.DataFrame(data=0, index=[0], columns = ["Puissance échangeur initiale en kW", "Nouvelle puissance échangeur en kW", "Débit de charge en L/h (en semi-instantané)", "Vitesse réseau de charge en m/s", "Puissance réchauffeur de boucle en kW"])
    dis["Puissance échangeur initiale en kW"] = Pech
    Pstock = Pech + pertes_bal
    
    if (qboucle ==0)&(type_bouclage != 'Aucun') :
        st.warning("Modifiez les paramètres du réseau de bouclage retour : le débit de bouclage n'est pas correct !")
        Prech = np.nan
        qcharge = np.nan
        vitesse_ch = np.nan
        dis["Nouvelle puissance échangeur en kW"] = Pech
        dis["Débit de charge en L/h (en semi-instantané)"] = qcharge
        dis["Vitesse réseau de charge en m/s"] = vitesse_ch
        dis["Puissance réchauffeur de boucle en kW"] = Prech
    
    
    if réchauffage== True :
        Pech = Pstock
        Prech = pertes_boucl
        qcharge = Pech*10**3/(Cp()*(tprod-tfroid))
        
    else :
        if système == "I" :
            Pech = Pstock + pertes_boucl
            qcharge = Pech*10**3/(Cp()*(tprod-tfroid))
            Prech = 0
        
        if (système == "SI")|(système == "ACC") :
    
            if type_bouclage == 'Aucun' :
                Pech = Pstock
                qcharge = Pech*10**3/(Cp()*(tprod-tfroid))
                Prech = 0

    
            if (type_bouclage=='tiers_sup_milieu')|(type_bouclage=='avant_ballon') :
                Pech = Pech + 27.3*pertes_boucl**2/Pech + pertes_bal
                qcharge = Pech*10**3/(Cp()*(tprod-tfroid))
                Prech = 0
             
            if (type_bouclage=='bas_stockage')|(type_bouclage=='avant_entree_ech') :
                Pech = Pech + 0.7*(qboucle)**0.5 + pertes_bal
                qcharge = Pstock*10**3/(Cp()*(tprod-tfroid))+qboucle
                Prech = 0
                #Tentree_ech = tprod - Pech*10**3/(Cp()*qcharge)
        
    vitesse_ch = qcharge*10**(-3)/(3600*pi*(Dint*10**(-3))**2/4)
    
    dis["Nouvelle puissance échangeur en kW"] = Pech
    dis["Débit de charge en L/h (en semi-instantané)"] = qcharge
    dis["Vitesse réseau de charge en m/s"] = vitesse_ch
    dis["Puissance réchauffeur de boucle en kW"] = Prech
        
    return dis


df_fin = Calcul_Pech_Qcharge(st.session_state.type_système, st.session_state.type_bouclage, st.session_state.réchauffage_indépendant, st.session_state.Pech, pertes_tot, pertes_tot_ball, Qbouclage, st.session_state.Tfroid, st.session_state.Tprod, st.session_state.Dint_charge)
st.dataframe(df_fin)
Pech_final = df_fin.loc[0, "Nouvelle puissance échangeur en kW"]
Prech = df_fin.loc[0, "Puissance réchauffeur de boucle en kW"]
Qcharge = df_fin.loc[0, "Débit de charge en L/h (en semi-instantané)"]

for k in df_fin.index :
    if (df_fin.loc[k]["Vitesse réseau de charge en m/s"]>2) :
        st.error("La vitesse dans le réseau de charge est trop importante, modifiez les paramètres !")
    else :
        st.success("La vitesse dans le réseau de charge est bien inférieure à 2 m/s.")
    
##Pompes

st.sidebar.header("Caractéristiques des pompes")
st.sidebar.number_input("SPP de la pompe de charge", value=200.0, key="SPP_charge")

df7 = pd.DataFrame(data=0, index=st.session_state.df.index, columns = ["Numéro de bouclage", "SPP de la pompe de bouclage"])
df7['Numéro de bouclage'] = st.session_state.df.index
gb = GridOptionsBuilder.from_dataframe(df7)
gb.configure_column('SPP de la pompe de bouclage', editable=True)
grid_options = gb.build()
grid_response = AgGrid(df7, gridOptions=grid_options, fit_columns_on_grid_load=True, update_mode = 'MODEL_CHANGED', data_return_mode='AS_INPUT')

SPP_df = grid_response['data']
SPP_df.index = [u for u in range(1, len(SPP_df.index)+1)]

def power_pump(SPP, debit) :
    ''' Calcule la puissance nécessaire à une pompe en fonction du débit et du SPP '''
    P = debit*10**(-3)/3600*SPP
    return P

st.markdown("> **Puissances des pompes**")
df8 = pd.DataFrame(data = [['Pompe de charge', str(power_pump(st.session_state.SPP_charge, df_fin.loc[0, "Débit de charge en L/h (en semi-instantané)"]))]],columns=["Pompe", "Puissance en W"])

nbouclage = st.session_state.df.shape[0]

for k in range(1, nbouclage+1) :
    nwrow = [('Pompe de bouclage '+str(k)), str(power_pump(SPP_df.loc[k, "SPP de la pompe de bouclage"], st.session_state.df6.loc[k,"Débit de bouclage (L/h)"]))]
    df8.loc[k, :] = nwrow
    
AgGrid(df8, fit_columns_on_grid_load = True)

st.markdown("## Côté primaire")

st.sidebar.header("Caractéristiques du réseau primaire")
st.sidebar.number_input("Longueur aller primaire (m)", min_value=0.0, value=50.0, key="L_aller_prim")
st.sidebar.number_input("Longueur retour primaire (m)", min_value=0.0, value=50.0, key="L_retour_prim")

st.sidebar.number_input("Coefficient aller de pertes thermiques par mètre linéique (W/(mK))", min_value=0.0, value=0.025, key="isol_aller_prim")
st.sidebar.number_input("Coefficient retour de pertes thermiques par mètre linéique (W/(mK))", min_value=0.0, value=0.025, key="isol_retour_prim")

st.sidebar.number_input("Rendement prod primaire/échangeur", min_value=0.0, max_value=1.0, value=0.9, key="rendement")

st.sidebar.number_input("Température aller primaire", min_value=0.0, value=65.0, key="Tfluid_aller_prim")
st.sidebar.number_input("Température retour primaire", min_value=0.0, value=55.0, key="Tfluid_retour_prim")

def losses_primaires(L1, isol1, tfluid1, L2, isol2, tfluid2, tamb) :
    losses = isol1*(tfluid1-tamb)*L1*10**(-3) + isol2*(tfluid2-tamb)*L2*10**(-3)
    return losses

def power_primaire(pertes, rendement, Pech) :
    return Pech/rendement + pertes


st.session_state.pertes_primaires = losses_primaires(st.session_state.L_aller_prim, st.session_state.isol_aller_prim, st.session_state.Tfluid_aller_prim, st.session_state.L_retour_prim, st.session_state.isol_retour_prim, st.session_state.Tfluid_retour_prim, st.session_state.Tambprim)
st.session_state.Pprim = power_primaire(st.session_state.pertes_primaires, st.session_state.rendement, Pech_final)
df9 = pd.DataFrame({'Pertes thermiques au primaire (kW)' : [st.session_state.pertes_primaires], "Puissance nécessaire au primaire (kW)" : [st.session_state.Pprim]})

AgGrid(df9, fit_columns_on_grid_load = True)

st.markdown("## Synthèse")

synthesis = pd.DataFrame({"Item" : ['Température eau froide, °C', 'Température eau chaude, °C', 'Puissance au primaire, kW', 'Puissance échangeur, kW', 'Puissance réchauffage de bouclage indépendant, kW', 'Volume de stockage, L', 'Débit de charge (SI), L/h', 'Puissance de la pompe de charge, W'], "Value" : [st.session_state.Tfroid, st.session_state.Tprod, st.session_state.Pprim, Pech_final, Prech, st.session_state.Vstock, Qcharge, df8.loc[0, 'Puissance en W']]})

nbouclage = st.session_state.df6.shape[0]

for k in range(1, nbouclage+1) :
    rows = synthesis.shape[0]
    nwrow = ["Débit de bouclage "+str(k)+", L/h", st.session_state.df6.loc[k, "Débit de bouclage (L/h)"]]
    nwrow2 = ["Puissance de la pompe de bouclage "+str(k), df8.loc[k, "Puissance en W"]]
    synthesis.loc[rows+1, :] = nwrow
    synthesis.loc[rows+2, :] = nwrow2
    
AgGrid(synthesis, fit_columns_on_grid_load = True)    