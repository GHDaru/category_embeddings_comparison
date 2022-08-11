
import streamlit as st
import time
import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import unidecode
from wordcloud import WordCloud
import matplotlib.pyplot as plt

################## Configurações da Página ###########################
st.set_page_config(page_title="Trabalho de Visualização",
				   page_icon=":shark:",
				   layout="wide")

#######################################
def get_category_vocabulary(df,category):
	tokens = set((' '.join(df[df['nm_product']==category]['nm_item'].values)).split())    
	return [unidecode.unidecode(str.upper(token)) for token in tokens]

def prepare_word_cloud(df, category):
	words = ' '.join(df[df['nm_product']==category]['nm_item'])
	wordcloud = WordCloud().generate(words)
	fig, ax = plt.subplots(figsize=(12, 12))
	ax.imshow(wordcloud, interpolation='bilinear')
	ax.axis("off")
	return fig
	
############Figuras############################
def plot_for_D(data, labels, need_labels, search_idx=None, filter_category = None, word01 = None, word02 = None):
	if dim=='2-D':
		linhas[3].text(f'Filtro plot D:{filter_category}')
		fig = plot_2D(data, labels, need_labels, search_idx, filter_category = filter_category, word01 = word01, word02=word02)        
		render_plot(fig)
	elif dim=='3-D':
		fig = plot_3D(data, labels, need_labels, search_idx)#, filter_category, word01, word02)
		render_plot(fig)
  
def render_plot(fig):
	fig.update_layout(margin={"r":50,"t":100,"l":0,"b":0}, height=750, width=850)
	linhas[3].plotly_chart(fig)


def plot_3D(data, labels, need_labels, search=None):
	sizes = [5]*len(labels)
	colors = ['rgb(93, 164, 214)']*len(labels)
	
	if search: 
		sizes[search] = 25
		colors[search] = 'rgb(243, 14, 114)'

	if not need_labels:
		labels=None

	fig = go.Figure(data=[go.Scatter3d(
			x=data[:,0], y=data[:,1], z=data[:,2],
			mode='markers+text',
			text=labels,
			marker=dict(
				color=colors,
				size=sizes
			)
		)], layout=Layout(
	paper_bgcolor='rgba(0,0,0,0)',
	plot_bgcolor='rgba(0,0,0,0)'))
	return fig
###########################################################

######################Carga de Dados#######################
@st.cache(allow_output_mutation=True)
def get_data():
	df_temp = pd.read_pickle('itens.pkl')
	df_temp.fillna('',inplace=True)
	categorias = df_temp.nm_product.unique()
	return df_temp[['nm_item','nm_product']].copy(),sorted(categorias)

@st.cache(allow_output_mutation=True)
def load_data():	
	file = "produtos_varejo.txt"		
	df = pd.read_csv(file, sep=';')
	data = df.iloc[:,1:].values
	labels = df.iloc[:,0].values	
	data_reduced = PCA(n_components=3).fit_transform(data)
	return data_reduced, labels

#Carga de Dados###############################
data, labels = load_data()
df, categorias = get_data()
######################## Contrução do layout ################################
st.header("Trabalho de Visualização de Informação - Word Embeddings")

linhas = [st.empty(), st.expander("Seletor de Categorias"), st.expander("Gráficos Gerado")]

############# Parâmetros #################
embeddings = ("Word2Vec Produtos Dimensão 100",)
embedding_type = st.sidebar.selectbox("Selecione Embutimento", embeddings)
dims = ("2-D", "3-D")
dim = st.sidebar.selectbox("Dimensions", dims)
col1, col2 = linhas[1].columns([.5,.5])
categoria01 = col1.selectbox("Categoria 01", categorias)#, format_func=lambda x: categorias[x], key='c1')
categoria02 = col2.selectbox("Categoria 02", categorias)#, format_func=lambda x: categorias[x], key='c2')
render_ok = categoria01 and categoria02
###########################################################

###########################################################
if render_ok:
	word01 = get_category_vocabulary(df,categoria01)
	col1.pyplot(prepare_word_cloud(df, categoria01))
	word02 = get_category_vocabulary(df,categoria02)
	col2.pyplot(prepare_word_cloud(df, categoria02))		
###############################################

if render_ok:    
	df_temp = pd.DataFrame({'labels':labels, 'x':data[:,0], 'y':data[:,1]}).reset_index()		
	df_temp.loc[:,'sizes'] = 5 
	df_temp.loc[:,'color'] = np.nan		
	df_temp.loc[df_temp.labels.isin(set(word01).difference(set(word02))),'color'] = f'apenas {categoria01}'
	df_temp.loc[df_temp.labels.isin(set(word02).difference(set(word01))),'color'] = f'apenas {categoria02}'
	df_temp.loc[df_temp.labels.isin(set(word01).intersection(set(word02))),'color'] = 'ambas'
	df_temp = df_temp[~df_temp.color.isna()]         
	data = data[df_temp['index'].values,:]
	fig = px.scatter(df_temp, 
			x='x',
   			y='y', 
			color='color',
			hover_data=['labels'],
		)
	linhas[2].plotly_chart(fig, use_container_width = True)