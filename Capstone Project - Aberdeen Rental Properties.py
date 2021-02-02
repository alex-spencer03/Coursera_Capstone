#!/usr/bin/env python
# coding: utf-8

# # Does proximity to amenities impacts the rental price for a property in Aberdeen?

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     
# 1. [A description of the problem and a discussion of the background](#0)<br>
# 2. [Data sources](#1)<br>
# 3. [Data scraping and handling null values](#2)<br>
# 4. [Data Analysis](#3)<br>
# 5. [Downloading and analysing the FourSquare data against rental properties](#4)<br>
# 6. [Conclusion](#5)<br>
# 
# </div>
# <hr>

# ## A description of the problem and a discussion of the background.<a id="0"></a>

# **Introduction/ Business Problem**
# 
# I have been renting in Aberdeen, Scotland for nearly 9 years and one thing I could never fully comprehend was the rental price. I would find 2 properties in close proximity to each other, and which to me seemed to be fairly similar, but with very different asking prices. 
# 
# It got me thinking: 'What actually determines the rental price for a property in Aberdeen?'. Is it the location? Maybe the square footage or EPC band? What about it's proximity to amenities such as shops, cafes and restaurants? There is surprisingly less data publicly available on this matter than compared to that accessible to home buyers.
# 
# As part of the IBM Data Science Professional Certificate, this project required the use of FourSquare data. The analysis therefore looks at the relationship between the rental prices of properties in Aberdeen and their proximity to amenities clusters.
# 
# **Target audience**
# 
# The conclusions of this mini-research are meant to give my fellow renters some insight into rental prices in Aberdeen and what most impacts them. This in turn is meant to aid them into deciding on what to focus in their searches.

# ## Data sources<a id="1"></a>

# The below data sources used for analysis have also been made available on my Github repository:
# https://github.com/alex-spencer03/Coursera_Capstone.
# 
# 1.	Data of rental properties available for rent: This was scraped from the Aberdeen Solicitors Property Centre’s (ASPC) website on the 15th July 2020: https://www.aspc.co.uk/.
# 2. Venue data from FourSquare's API
# 
# A full explanation of each dataset and features can be found in the final report available via the above GitHub link.

# ## Data scraping and handling null values<a id="2"></a>

# In[1]:


# Importing required libraries

import pandas as pd
from pandas.io.json import json_normalize
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import numpy as np
import json
import io
import requests	 # library to handle requests
import re  #very useful for defining search patterns

# Matplotlib and associated plotting modules
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as colors
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.cluster import KMeans

#!pip install geopy
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

#!pip install folium
import folium # map rendering library

print('Libraries imported.')


# In[2]:


url = 'https://www.aspc.co.uk/search/?PrimaryPropertyType=Rent&SortBy=PublishedDesc&LastUpdated=AddedAnytime&SearchTerm=&PropertyType=Residential&PriceMin=&PriceMax=&Bathrooms=&OrMoreBathrooms=true&Bedrooms=&OrMoreBedrooms=true&HasCentralHeating=false&HasGarage=false&HasDoubleGarage=false&HasGarden=false&IsNewBuild=false&IsDevelopment=false&IsParkingAvailable=false&IsPartExchangeConsidered=false&PublicRooms=&OrMorePublicRooms=true&IsHmoLicense=false&IsAllowPets=false&IsAllowSmoking=false&IsFullyFurnished=false&IsPartFurnished=false&IsUnfurnished=false&ExcludeUnderOffer=false&IncludeClosedProperties=true&ClosedDatesSearch=14&MapSearchType=EDITED&ResultView=LIST&ResultMode=NONE&AreaZoom=13&AreaCenter[lat]=57.14955426557916&AreaCenter[lng]=-2.0927401123046785&EditedZoom=13&EditedCenter[lat]=57.14955426557916&EditedCenter[lng]=-2.0927401123046785'
api_url = 'https://api.aspc.co.uk/Property/GetProperties?{}&Sort=PublishedDesc&Page=1&PageSize=639'

params = url.split('?')[-1]
data = requests.get(api_url.format(params)).json()

#print(json.dumps(data, indent=4))


# In[3]:


# creating the initial pandas dataframe
column_names=['First_line','Second_line','City','Postcode','Bedrooms','Bathrooms','Lounges','Price','Square_ft','Property_type','Description','Coordinates']
properties = pd.DataFrame(columns=column_names)
properties


# In[4]:


# pulling the required info for each column
for property_ in data:
    first_line=property_['Location']['AddressLine1']
    second_line=property_['Location']['LineTwoLocation']
    city=property_['Location']['City']
    postcode=property_['Location']['Postcode']
    bedrooms=property_['Bedrooms']
    bathrooms=property_['Bathrooms']
    lounges=property_['PublicRooms']
    price=property_['Price']
    square_ft=property_['FloorArea']
    property_type=property_['PropertyIconKey']
    description=property_['CategorisationDescription']
    coordinates=property_['Location']['Spatial']['Geography']['WellKnownText']
    
# amending the existing dataframe with required info
    properties=properties.append({'First_line': first_line,
                                 'Second_line':second_line,
                                 'City': city,
                                 'Postcode': postcode,
                                 'Bedrooms': bedrooms,
                                 'Bathrooms': bathrooms,
                                 'Lounges': lounges,
                                 'Price': price,
                                 'Square_ft': square_ft,
                                 'Property_type': property_type,
                                 'Description': description,
                                 'Coordinates': coordinates}, ignore_index=True)

properties.head()


# In[5]:


properties.shape


# ### Data cleaning and handling null values

# In[6]:


import re  #very useful for defining search patterns

# creating the Council tax, EPC band, Garden and Parking columns by extracting the data from the Description column
# note to self: (?i) case insensitive modifier

properties['Council_tax_band'] = properties['Description'].str.extract(r'(?i)\(((?:CT)[^()*&?%]+)\)', expand=False)
properties['EPC_band'] = properties['Description'].str.extract(r'(?i)\(((?:EPC)[^()*&?%]+)\)', expand=False)
properties['Garden']= properties.Description.str.extract(r'\b(Garden)\b', expand=False)
properties['Parking']= properties.Description.str.extract(r'\b(Parking)\b', expand=False)

#sorting out the latitude and longitude of the properties
properties['Coordinates']=properties.Coordinates.str.strip('POINT (')
properties['Coordinates']=properties.Coordinates.str.strip(')')
properties['Longitude'], properties['Latitude']= properties['Coordinates'].str.split(' ',1).str

#dropping columns no longer required
properties.drop(['Description'], axis=1, inplace=True)
properties.drop(['Second_line'], axis=1, inplace=True)
properties.drop(['Coordinates'], axis=1, inplace=True)

properties.head()


# In[7]:


properties['Council_tax_band'].value_counts()


# ### Because the `Council_tax_band` strings pulled from ASPC's API are not consistent further work is required on them.

# In[8]:


#striping down council tax data
properties['Council_tax_band']=properties.Council_tax_band.str.rstrip()
properties['Council_tax_band']=properties.Council_tax_band.str.replace('band', 'Band')
properties['Council_tax_band']=properties.Council_tax_band.str.replace('CT - Band TBC','CT Band - TBC')
properties['Council_tax_band']=properties.Council_tax_band.str.replace('CT BandTBC','CT Band - TBC')
properties['Council_tax_band']=properties.Council_tax_band.str.replace('CT Band -E','CT Band - E')
properties['Council_tax_band']=properties.Council_tax_band.str.replace('CT Band -F','CT Band - F')
properties['Council_tax_band']=properties.Council_tax_band.str.replace(' -','')

properties['EPC_band']=properties.EPC_band.str.replace(' -','')

#making property_type entries more pleasent to view
properties['Property_type']=properties.Property_type.str.replace('FLAT', 'Flat')
properties['Property_type']=properties.Property_type.str.replace('HOUSE_SEMI_DETACHED', 'House Semi Detached')
properties['Property_type']=properties.Property_type.str.replace('HOUSE_DETACHED', 'House Detached')
properties['Property_type']=properties.Property_type.str.replace('HOUSE_TERRACED', 'House Terraced')

properties.head()


# In[9]:


#sanity check
properties['Council_tax_band'].value_counts()


# #### Much better

# In[10]:


aberdeen=properties[properties['City'] == 'Aberdeen'].reset_index(drop=True)
aberdeen.head()


# In[11]:


aberdeen_master=aberdeen[aberdeen['City'] == 'Aberdeen'].reset_index(drop=True)
aberdeen_master.shape


# ### Now that we have a set dataframe we need to investigate and handle any null values which will impact the analysis

# In[12]:


#checking the entire dataframe for null values
aberdeen_master.isnull().sum()


# In[13]:


#checking which properties do not have a council tax entry
aberdeen_master[aberdeen_master['Council_tax_band'].isnull()]


# In[14]:


#checking which properties do not have an EPC entry
aberdeen_master[aberdeen_master['EPC_band'].isnull()]


# In[15]:


#checking properties with no Square_ft data
aberdeen_master.loc[aberdeen_master['Square_ft'] == 0]


# In[16]:


prop_sqft=pd.DataFrame((aberdeen_master.loc[aberdeen_master['Square_ft'] > 0])).reset_index(drop=True)
prop_sqft[['Price','Square_ft']].corr()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


sns.regplot(x="Square_ft", y="Price", data=prop_sqft)
plt.ylim(0,)


# ### Important: this data was initially pulled on 15th July 2020 and the above code was for illustrative purposes only.
# 
# Just as can be seen now the number of properties without square footage values is high. To obtain this data would have been very time consuming and I had to deternine if it had any value to bring to justify the time and resources to obtain it.
# 
# As can be seen in the above .corr() function there is a 0.851765 correlation between the price and the square footage of the property. The above scatter plot confirms this. Not including this data could have skewed the final results.
# 
# The initial properties dataset was persisted to DB2 Warehouse on 15th July 2020 and uploaded to GitHub on 5th August 2020.

# ## Data Analysis<a id="3"></a>

# In[19]:


import pandas as pd
import io

# Downloading csv file from my GitHub account
url2 = "https://raw.githubusercontent.com/alex-spencer03/Coursera_Capstone/master/aberdeen_master.csv"
download = requests.get(url2).content

# Reading doanloded content into pandas dataframe
aberdeen_master = pd.read_csv(io.StringIO(download.decode('utf-8')))

# sanity check
aberdeen_master.head()


# In[20]:


#sanity check
print(aberdeen_master.dtypes)


# In[21]:


# Dropping City attribute as all instances have the same city
aberdeen_master.drop(['City'], axis=1, inplace=True)


# In[22]:


aberdeen_master.describe()


# #### Let's calculate and then visualise the correlation between the numerical variables

# In[23]:


aberdeen_master[['Price','Bedrooms','Bathrooms','Lounges','Square_ft','Parking']].corr()


# In[24]:


sns.regplot(x='Bedrooms', y='Price', data=aberdeen_master)
plt.ylim(0,)


# In[25]:


sns.regplot(x='Bathrooms', y='Price', data=aberdeen_master)
plt.ylim(0,)


# In[26]:


sns.regplot(x='Lounges', y='Price', data=aberdeen_master)
plt.ylim(0,)


# In[27]:


sns.regplot(x='Square_ft', y='Price', data=aberdeen_master)
plt.ylim(0,)


# #### Calculating and visualising the correlation between the categorical variables

# In[28]:


ax=sns.boxplot(x="Council_tax_band", y="Price", data=aberdeen_master)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# In[29]:


ax=sns.boxplot(x="EPC_band", y="Price", data=aberdeen_master)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# In[30]:


ax=sns.boxplot(x="Council_ward", y="Price", data=aberdeen_master)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


# In[31]:


#applying one-hot encoding to the categorical variables for further analysis
for column in ['Property_type','Council_tax_band','EPC_band','Parking']:
    dummies = pd.get_dummies(aberdeen_master[column])
    aberdeen_master[dummies.columns] = dummies


# In[32]:


aberdeen_master[['Price','CT Band A', 'CT Band B', 'CT Band C', 'CT Band D', 'CT Band E', 'CT Band F', 'CT Band G', 'CT Band H']].corr()


# In[33]:


aberdeen_master[['Price','EPC band B','EPC band C','EPC band D','EPC band E','EPC band F','EPC band G']].corr()


# #### Let's investigate a bit more the relationship between the different variables by looking at the average prices

# In[34]:


test_one=aberdeen_master[['Bedrooms','Bathrooms','Lounges','Square_ft','Price']]
ab_avg_price=test_one.groupby(['Bedrooms','Bathrooms','Lounges'], as_index=False).mean()
ab_avg_price['Sq_diff']=ab_avg_price['Square_ft'].diff()
ab_avg_price['Price_diff']=ab_avg_price['Price'].diff()

#ab_avg_price.sort_values(by=['Difference'], inplace=True, ascending=False)
ab_avg_price


# #### Now let's see what the average rental property looks like in each council ward

# In[35]:


test_one=aberdeen_master[['Council_ward','Bedrooms','Bathrooms','Lounges','Square_ft','Price']]
c_ward_avg=test_one.groupby(['Council_ward'], as_index=False).mean()
c_ward_avg.sort_values(by=['Price'], inplace=True, ascending=False)
c_ward_avg


# #### Let's cluster the properties and visualise the clusters

# In[36]:


# import k-means from clustering stage
from sklearn.cluster import KMeans

# set number of clusters
kclusters = 3

aberdeen_grouped_clustering = aberdeen_master[['Bedrooms','Bathrooms','Lounges','Square_ft','Price','Latitude','Longitude']]

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(aberdeen_grouped_clustering)

# add clustering labels
aberdeen_grouped_clustering.insert(0, 'Cluster Labels', kmeans.labels_)
aberdeen_grouped_clustering.head()


# In[37]:


import numpy as np # library to handle data in a vectorized manner

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#!pip install geopy
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[38]:


#!pip install folium


# In[39]:


import folium # map rendering library

print('Libraries imported.')


# In[40]:


#adding the council_wards into dataframe
council_w=aberdeen_master[['Council_ward','Latitude']]
aberdeen_grouped_clustering=pd.merge(aberdeen_grouped_clustering, council_w, on='Latitude')

# move neighborhood column to the first column
cols = [aberdeen_grouped_clustering.columns[-1]] + list(aberdeen_grouped_clustering.columns[:-1])
aberdeen_grouped_clustering = aberdeen_grouped_clustering[cols]

aberdeen_grouped_clustering.head()


# In[41]:


#getting the coordinates for Aberdeen city
address = 'Aberdeen, Scotland'

geolocator = Nominatim(user_agent="ab_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Aberdeen, Scotland are {}, {}.'.format(latitude, longitude))


# In[42]:


# creating the map of clusters
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# setting the color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# adding the markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(aberdeen_grouped_clustering['Latitude'], aberdeen_grouped_clustering['Longitude'], aberdeen_grouped_clustering['Council_ward'], aberdeen_grouped_clustering['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# #### Finally, let's examine each cluster

# In[43]:


#Examining the first cluster of properties
cluster_one=pd.DataFrame(aberdeen_grouped_clustering.loc[aberdeen_grouped_clustering['Cluster Labels'] == 0])
cluster_one.describe()


# In[44]:


#Examining the second cluster of properties
cluster_two=pd.DataFrame(aberdeen_grouped_clustering.loc[aberdeen_grouped_clustering['Cluster Labels'] == 1])
cluster_two.describe()


# In[45]:


#Examining the third and final cluster of properties
cluster_three=pd.DataFrame(aberdeen_grouped_clustering.loc[aberdeen_grouped_clustering['Cluster Labels'] == 2])
cluster_three.describe()


# ## Downloading and analysing the FourSquare data against rental properties<a id="4"></a>

# In[46]:


CLIENT_ID = 'EXSW1MIZO4ZE4VQQP3ZW0GN3KCNDP5ZEWYIC25EUOTMG5Y24' # your Foursquare ID
CLIENT_SECRET = 'VTQFVGSCC0H5GWDIWSZK3RQRHICOF0JB2MHS4CA5ROTLQOTB' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[47]:


LIMIT = 700 # limit of number of venues returned by Foursquare API
radius = 7000 # define radius
# create URL
url3 = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    latitude, 
    longitude, 
    radius, 
    LIMIT)

url3 # display URL


# In[48]:


results = requests.get(url3).json()
results


# In[49]:


venues = results['response']['groups'][0]['items']
    
ab_venues = pd.json_normalize(venues) # flatten JSON

# filtering the columns
filtered_columns = ['venue.name', 'venue.location.postalCode','venue.location.lat', 'venue.location.lng']
ab_venues =ab_venues.loc[:, filtered_columns]

# cleaning the columns
ab_venues.columns = [col.split(".")[-1] for col in ab_venues.columns]

#renaming the columns
ab_venues=ab_venues.rename(columns={'lat':'Latitude','lng':'Longitutde','name':'Venue_name','postalCode':'PostCode'})

ab_venues.head()


# In[50]:


print('{} venues were returned by Foursquare.'.format(ab_venues.shape[0]))


# In[51]:


# creating a map of Aberdeen venues returned from FourSquare
ab_venues_map = folium.Map(location=[latitude, longitude], zoom_start=11)

# adding markers to map
for lat, lng, label in zip(ab_venues['Latitude'], ab_venues['Longitutde'], ab_venues['Venue_name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(ab_venues_map)  
    
ab_venues_map


# ## Conclusion<a id="5"></a> 

# As can be seen when comparing the two maps, the pattern of rental price does not follow the one of venues' frequency. Therefore, proximity to amenities does not impact a rental property's asking price.

# # Thank you for your time.
