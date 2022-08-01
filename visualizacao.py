
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
####Variáveis##########
word01 = [0]
word02 = [0]


################## Configurações da Página ###########################
st.set_page_config(page_title="Trabalho de Visualização",
				   page_icon=":shark:",
				   layout="wide")

def title():
	# header
	st.markdown("####  Trabalho de Visualização de Informação - Word Embeddings")
	return

title()

col1, col2, col3 = st.columns([30,5,5])
#######################################
def get_category_vocabulary(df,category):
	tokens = set((' '.join(df[df['nm_product']==category]['nm_item'].values)).split())    
	return [unidecode.unidecode(str.upper(token)) for token in tokens]
	
############Figuras############################
def plot_for_D(data, labels, need_labels, search_idx=None, filter_category = None, word01 = None, word02 = None):
	if dim=='2-D':
		col3.text(f'Filtro plot D:{filter_category}')
		fig = plot_2D(data, labels, need_labels, search_idx, filter_category = filter_category, word01 = word01, word02=word02)        
		render_plot(fig)
	elif dim=='3-D':
		fig = plot_3D(data, labels, need_labels, search_idx)#, filter_category, word01, word02)
		render_plot(fig)

def plot_2D(data, labels, need_labels, search=None, filter_category = False, word01 = None, word02 = None):
	col3.text(f'Filtro:{filter_category}')
	df_temp = pd.DataFrame({'labels':labels}).reset_index()	
	col3.text(df_temp.shape)	
	df_temp.loc[:,'sizes'] = 5 
	df_temp.loc[:,'color'] = np.nan
	col3.dataframe(df_temp)      
	if filter_category:
		df_temp.loc[df_temp.labels.isin(set(word01).difference(set(word02))),'color'] = 'rgb(0, 0, 255)'
		df_temp.loc[df_temp.labels.isin(set(word02).difference(set(word01))),'color'] = 'rgb(255, 0, 0)'
		df_temp.loc[df_temp.labels.isin(set(word01).intersection(set(word02))),'color'] = 'rgb(0, 255, 0)'
		df_temp = df_temp[~df_temp.color.isna()]         
		data = data[df_temp['index'].values,:]
	else:
		aleatorio = np.random.randint(0, len(data),1000)
		df_temp = df_temp.iloc[aleatorio]
		df_temp.loc[:,'color'] = 'rgb(93, 164, 214)'
		data = data[aleatorio,:]
		
	# if search: 
	#     sizes[search] = 25
	#     colors[search] = 'rgb(243, 14, 114)'
		
	# if not need_labels:
	#     labels=None
	fig = go.Figure(data=[go.Scatter(
			x=data[:,0], y=data[:,1],
			mode='markers+text',
			text=df_temp.labels.values if need_labels else None,
			marker=dict(
				color=df_temp.color.values,
				size=df_temp.sizes.values,
			)
		)],layout=Layout(
	paper_bgcolor='rgba(0,0,0,0)',
	plot_bgcolor='rgba(0,0,0,0)'))
	return fig

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
	categorias = df_temp.nm_product.unique()
	return df_temp[['nm_item','nm_product']].copy(),categorias
#def get_df():
	# with connect_azure_read() as conn:
	#     df = pd.read_sql('SELECT tt.*, nm_product FROM sandbox.tbl_saneamento_teste tt LEFT JOIN tbl_product tp ON tp.id_product = tt.id_product', conn)
	#     product = pd.read_sql('SELECT id_product, nm_product FROM tbl_product', conn)
	#     df = df.mer
df, categorias = get_data()

###########################################################

##############Side Bar########
#############Dados#################
embeddings = ("Word2Vec Produtos Dimensão 100",)
options = list(range(len(embeddings)))
embedding_type = st.sidebar.selectbox("Selecione Embutimento", options, format_func=lambda x: embeddings[x])
st.sidebar.text('OR')
uploaded_file = st.sidebar.file_uploader("Enviar arquivo(Optional)", type="txt")

def load_data(embedding_type):
	if embedding_type==0:
		file = "produtos_varejo.txt"		
	df = pd.read_csv(file, sep=';')
	data = df.iloc[:,1:].values
	labels = df.iloc[:,0].values	
	data_reduced = PCA(n_components=3).fit_transform(data)
	return data_reduced, labels

if not uploaded_file:
	data, labels = load_data(embedding_type)
else:
	df = pd.read_csv(upload_file, sep=';')
	data = df.iloc[:,1:].values
	labels = df.iloc[:,0].values	
	data = PCA(n_components=3).fit_transform(data)
	
	
 
###############Opções############################
# # def display_reductions():
# # 	reductions = ("PCA", "TSNE")
# # 	options = list(range(len(reductions)))
# # 	reductions_type = st.sidebar.selectbox("Selecione o Tipo de Redução", options, format_func=lambda x: reductions[x])
# # 	return reductions_type
# reductions_type = display_reductions()

def display_dimensions():
	dims = ("2-D", "3-D")
	dim = st.sidebar.radio("Dimensions", dims)	
	return dim

dim = display_dimensions()

# search
def display_search():
	search_for = st.sidebar.text_input("Palavra para Realçar", "")
	return search_for
search_for = display_search()

#labels check
def display_labels():
	need_labels = st.sidebar.checkbox("Display Labels", value=True)
	return need_labels
need_labels = display_labels()

button = st.sidebar.button('Visualise')

###############################################

filter_category = col2.checkbox("Comparar categorias", value=False)
col2.text(filter_category)
if filter_category:
	options = list(range(len(categorias)))
	categoria01 = col2.selectbox("Categoria ", options, format_func=lambda x: categorias[x], key='c1')
	col2.text(categoria01)
	categoria02 = col2.selectbox("Categoria ", options, format_func=lambda x: categorias[x], key='c2')
	col2.text(categoria02)
	word01 = get_category_vocabulary(df,categorias[categoria01])
	word02 = get_category_vocabulary(df,categorias[categoria02])
	col2.text(word01[:5])
	col2.text(word02[:5])
	
###############################################

def render_plot(fig):
	fig.update_layout(margin={"r":50,"t":100,"l":0,"b":0}, height=750, width=850)
	col1.plotly_chart(fig)

if button:	    
	col2.text(f'Word:{word01[0]}')
	if search_for:
		search_idx = labels.index(search_for)        
		plot_for_D(data, labels, need_labels, search_idx, filter_category, word01, word02)
	else:
		col3.text('Nenhum Palavra Selecionada')
		col3.text(f'Filtro:{filter_category}')
		plot_for_D(data, labels, need_labels,filter_category=filter_category, word01 = word01, word02=word02)


	
	





