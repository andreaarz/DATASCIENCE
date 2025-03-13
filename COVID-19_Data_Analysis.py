import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Cargar datos de COVID-19 (puedes descargar el archivo desde https://github.com/CSSEGISandData/COVID-19)
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
data = pd.read_csv(url)

# Mostrar las primeras filas del dataset
print(data.head())

# Limpiar y preparar los datos
# Agrupar los datos por país (suma de los casos confirmados por cada día)
data_long = data.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], 
                     var_name="Date", value_name="Confirmed Cases")
data_long['Date'] = pd.to_datetime(data_long['Date'])

# Filtrar datos para un país específico (por ejemplo, "United States")
us_data = data_long[data_long['Country/Region'] == 'US']

# Crear una gráfica de líneas interactiva con Plotly para mostrar la evolución de los casos confirmados en EE.UU.
fig = px.line(us_data, x='Date', y='Confirmed Cases', title='COVID-19 Confirmed Cases in the United States')
fig.update_xaxes(rangeslider_visible=True)
fig.show()

# También puedes comparar varios países (por ejemplo, Estados Unidos y España)
countries_of_interest = ['US', 'Spain', 'Italy', 'Germany']
filtered_data = data_long[data_long['Country/Region'].isin(countries_of_interest)]

# Gráfico interactivo comparando la evolución de casos confirmados en diferentes países
fig_comparison = px.line(filtered_data, x='Date', y='Confirmed Cases', color='Country/Region', 
                         title='COVID-19 Confirmed Cases Comparison Across Countries')
fig_comparison.update_xaxes(rangeslider_visible=True)
fig_comparison.show()

# Gráfico adicional de barras con Matplotlib para mostrar los casos confirmados totales por país
total_cases_by_country = data_long.groupby('Country/Region')['Confirmed Cases'].max().sort_values(ascending=False)
top_10_countries = total_cases_by_country.head(10)

# Gráfico de barras
plt.figure(figsize=(10, 6))
top_10_countries.plot(kind='bar', color='skyblue')
plt.title('Top 10 Countries with Most Confirmed COVID-19 Cases')
plt.xlabel('Country')
plt.ylabel('Confirmed Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

""" Explicación:
Cargar los datos:

Se descarga el archivo CSV que contiene los datos de COVID-19 globales desde el repositorio de Johns Hopkins University.
Preparación de los datos:

Se transforma el conjunto de datos a un formato largo (de "wide" a "long"), donde las columnas representan las fechas y los valores representan los casos confirmados.
Visualización con Plotly:

Se crea una visualización interactiva de las tendencias de los casos confirmados en los Estados Unidos. La visualización permite hacer zoom y ver más detalles con el rangeslider de Plotly.
Además, se compara la evolución de los casos en varios países (Estados Unidos, España, Italia y Alemania).
Visualización con Matplotlib:

Se genera un gráfico de barras para visualizar los 10 países con más casos confirmados de COVID-19 en el momento más reciente disponible en el conjunto de datos.
Resultado:
Gráfico de Líneas Interactivo: Muestra cómo han evolucionado los casos de COVID-19 en los Estados Unidos a lo largo del tiempo.
Comparación de Países: Compara la evolución de los casos en varios países de forma interactiva.
Gráfico de Barras: Muestra los 10 países con más casos confirmados hasta la fecha.
Key Takeaways:
Interactividad: Se utiliza Plotly para crear gráficos interactivos que permiten explorar los datos de forma dinámica.
Análisis Comparativo: Se compara la evolución de los casos en diferentes países para ver cómo las medidas de control han influido en los resultados.
Visualización Claras: Se utiliza Matplotlib para mostrar de forma clara y sencilla los países con más casos confirmados.

"""
