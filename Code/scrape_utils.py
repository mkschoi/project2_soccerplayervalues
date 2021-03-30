import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import time, os
import requests


def standard_table(URL):
    '''
    input: URL of FBREF website containing soccer player stats
    output: a dataframe of the standard stats for all senior players in the top 5 European leagues 
    '''

    # URL html for market value table read in beautiful soup
    page = requests.get(URL).text
    soup = BeautifulSoup(page, features='lxml')
    
    # Finding standard stats table
    standard_table = soup.find('table', id='stats_standard')

    # Put player names in list
    players = [header for header in standard_table.find_all('td', {'class':'left','data-stat':'player'})]
    player_names = [name.text for th in players for name in th.find_all('a')]

    # Put player stats in list
    player_row = [row for row in standard_table.find_all('tr', class_= lambda x: x != 'thead')]

    player_stats = {}

    for player in player_row[2:]: 
        player_stats_list = []
        
        player_stats_list.append(player.find('td',{'data-stat':'comp_level'}).text)        
        player_stats_list.append(player.find('td',{'data-stat':'position'}).text)
        player_stats_list.append(player.find('td',{'data-stat':'age'}).text)
        player_stats_list.append(int(player.find('td',{'data-stat':'games'}).text))
        player_stats_list.append(int(player.find('td',{'data-stat':'games_starts'}).text))
        player_stats_list.append(float(player.find('td',{'data-stat':'minutes_90s'}).text))
        player_stats_list.append(int(player.find('td',{'data-stat':'goals'}).text))
        player_stats_list.append(int(player.find('td',{'data-stat':'assists'}).text))
        player_stats_list.append(int(player.find('td',{'data-stat':'cards_yellow'}).text))
        player_stats_list.append(int(player.find('td',{'data-stat':'cards_red'}).text))
        
        player_stats[player] = player_stats_list
        
    player_stats_list_all = [stats for stats in player_stats.values()]

    # Creating dataframe for players and stats
    player_stats_df = pd.DataFrame(player_stats_list_all)
    player_stats_df.columns = ['League', 'Position', 'Age', 'Matches Played', 'Starts', '90s Played',
       'Goals', 'Assists', 'Yellow Cards', 'Red Cards']
    player_stats_df.index = player_names

    return player_stats_df

def convert_league(string):
    if len(string) >= 1:
        return ' '.join(string.split(' ')[1:])
    else:
        return np.nan

def convert_int(string):
    if len(string) >= 1:
        return int(string)
    else:
        return np.nan
    
def shooting_table(URL):
    '''
    input: URL of FBREF website containing soccer player stats
    output: dataframe of the shooting stats for all senior players in the top 5 European leagues 
    '''

    # URL html for market value table read in beautiful soup
    page = requests.get(URL).text
    soup = BeautifulSoup(page, features='lxml')
    
    # Finding standard stats table
    shooting_table = soup.find('table', id='stats_shooting')

    # Put player names in list
    players = [header for header in shooting_table.find_all('td', {'class':'left','data-stat':'player'})]
    player_names = [name.text for th in players for name in th.find_all('a')]

    # Put player stats in list
    player_row = [row for row in shooting_table.find_all('tr', class_= lambda x: x != 'thead')]

    player_stats = {}

    for player in player_row[2:]: 
        player_stats_list = []
        
        player_stats_list.append(player.find('td',{'data-stat':'shots_total'}).text)        
        player_stats_list.append(int(player.find('td',{'data-stat':'shots_on_target'}).text))
        
        player_stats[player] = player_stats_list
        
    player_stats_list_all = [stats for stats in player_stats.values()]

    # Creating dataframe for players and stats
    player_stats_df = pd.DataFrame(player_stats_list_all)
    player_stats_df.columns = ['Total Shots', 'Shots on Target']
    player_stats_df.index = player_names

    return player_stats_df

def passing_table(URL):
    '''
    input: URL of FBREF website containing soccer player stats
    output: dataframe of the passing stats for all senior players in the top 5 European leagues 
    '''

    # URL html for market value table read in beautiful soup
    page = requests.get(URL).text
    soup = BeautifulSoup(page, features='lxml')
    
    # Finding standard stats table
    passing_table = soup.find('table', id='stats_passing')

    # Put player names in list
    players = [header for header in passing_table.find_all('td', {'class':'left','data-stat':'player'})]
    player_names = [name.text for th in players for name in th.find_all('a')]

    # Put player stats in list
    player_row = [row for row in passing_table.find_all('tr', class_= lambda x: x != 'thead')]

    player_stats = {}

    for player in player_row[2:]: 
        player_stats_list = []
        
        player_stats_list.append(player.find('td',{'data-stat':'passes_completed'}).text)        
        player_stats_list.append(player.find('td',{'data-stat':'passes'}).text)
        player_stats_list.append(player.find('td',{'data-stat':'assisted_shots'}).text)
        player_stats_list.append(player.find('td',{'data-stat':'passes_into_penalty_area'}).text)
        
        player_stats[player] = player_stats_list
        
    player_stats_list_all = [stats for stats in player_stats.values()]

    # Creating dataframe for players and stats
    player_stats_df = pd.DataFrame(player_stats_list_all)
    player_stats_df.columns = ['Passes Completed', 'Passes Attempted', 'Key Passes', 'Completed Passes into PA']
    player_stats_df.index = player_names

    return player_stats_df

def possession_table(URL):
    '''
    input: URL of FBREF website containing soccer player stats
    output: dataframe of the possession stats for all senior players in the top 5 European leagues 
    '''

    # URL html for market value table read in beautiful soup
    page = requests.get(URL).text
    soup = BeautifulSoup(page, features='lxml')
    
    # Finding standard stats table
    possession_table = soup.find('table', id='stats_possession')

    # Put player names in list
    players = [header for header in possession_table.find_all('td', {'class':'left','data-stat':'player'})]
    player_names = [name.text for th in players for name in th.find_all('a')]

    # Put player stats in list
    player_row = [row for row in possession_table.find_all('tr', class_= lambda x: x != 'thead')]

    player_stats = {}

    for player in player_row[2:]: 
        player_stats_list = []
        
        player_stats_list.append(player.find('td',{'data-stat':'players_dribbled_past'}).text)        
        player_stats_list.append(player.find('td',{'data-stat':'carries_into_penalty_area'}).text)
        
        player_stats[player] = player_stats_list
        
    player_stats_list_all = [stats for stats in player_stats.values()]

    # Creating dataframe for players and stats
    player_stats_df = pd.DataFrame(player_stats_list_all)
    player_stats_df.columns = ['Players Dribbled Past', 'Dribbles into PA']
    player_stats_df.index = player_names

    return player_stats_df

def market_value_table(URL):
    '''
    input: URL of transfermarkt.us containing player market value data
    output: dataframe of the market values for all senior players in the top 5 European leagues 
    '''
    # Allows access to the transfermrkt.us website for web scrapping
    headers = {'User-Agent': 
               'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
    
    # URL html for market value table read in beautiful soup
    page = requests.get(URL, headers = headers)
    soup = BeautifulSoup(page.content, features='lxml')
    
    # Player market values in list
    player_row = [row for row in soup.find('div',id='yw1').find_all('tr', class_=['odd','even'])]
    
    player_market_value = {}
    for player in player_row:
        player_name = player.find('td', class_='hauptlink').text.strip()
        player_market_value[player_name] = player.find('td', class_=['rechts hauptlink','rechts hauptlink mwHoechstwertKarriere']).text.strip()
    
    # Create a dataframe for player names and market values
    market_values_df = pd.DataFrame(player_market_value, index=[0]).T.reset_index() 
    market_values_df.columns = ['Player Name','Current Market Value']
    
    # Data cleaning
    def convert_market_value(value):
        list_value = re.split('(\d+)', value)
        if list_value[-1] == 'm':
            return float(''.join(list_value[1:4]))*1000000
        elif list_value[-1] == 'Th.':
            return int(list_value[1])*1000
        
    market_values_df['Current Market Value (USD)'] = market_values_df['Current Market Value'].apply(convert_market_value)
    market_values_df.drop('Current Market Value', axis=1, inplace=True)
    
    return market_values_df

def concat_table_epl():
    '''
    Concentenates EPL player market value tables that span multiple pages on the website
    '''
    appended_data = []
    for i in range(24):
        URL_page = 'https://www.transfermarkt.us/premier-league/marktwertaenderungen/wettbewerb/GB1/page/{}'.format(i)
        data = market_value_table(URL_page)
        appended_data.append(data)
        
    return pd.concat(appended_data)

def concat_table_laliga():
    '''
    Concentenates La Liga player market value tables that span multiple pages on the website
    '''
    appended_data = []
    for i in range(22):
        URL_page = 'https://www.transfermarkt.us/primera-division/marktwertaenderungen/wettbewerb/ES1/page/{}'.format(i)
        data = market_value_table(URL_page)
        appended_data.append(data)
        
    return pd.concat(appended_data)

def concat_table_seriea():
    '''
    Concentenates Serie A player market value tables that span multiple pages on the website
    '''
    appended_data = []
    for i in range(25):
        URL_page = 'https://www.transfermarkt.us/serie-a/marktwertaenderungen/wettbewerb/IT1/page/{}'.format(i)
        data = market_value_table(URL_page)
        appended_data.append(data)
        
    return pd.concat(appended_data)

def concat_table_bundesliga():
    '''
    Concentenates Bundesliga player market value tables that span multiple pages on the website
    '''
    appended_data = []
    for i in range(22):
        URL_page = 'https://www.transfermarkt.us/1-bundesliga/marktwertaenderungen/wettbewerb/L1/page/{}'.format(i)
        data = market_value_table(URL_page)
        appended_data.append(data)
        
    return pd.concat(appended_data)

def concat_table_ligue1():
    '''
    Concentenates Ligue 1 player market value tables that span multiple pages on the website
    '''
    appended_data = []
    for i in range(24):
        URL_page = 'https://www.transfermarkt.us/ligue-1/marktwertaenderungen/wettbewerb/FR1/page/{}'.format(i)
        data = market_value_table(URL_page)
        appended_data.append(data)
        
    return pd.concat(appended_data)
