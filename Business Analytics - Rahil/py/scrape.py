# scraping functions are not in use at the moment

from selenium import webdriver
import os

def get_driver(file_name, switch=1):
    if switch == 1:
        driver = webdriver.Chrome(os.getcwd() + file_name)
        return driver

def game_scrape(tt, game_data, driver):
    games = game_data.Game_ID.unique()

    for game in games[0:4]:
        qtr_scores_array = []
        driver.get('https://stats.nba.com/game/00' + str(game))
        table_rows = driver.find_elements_by_xpath("//div[@class='game-summary-linescore']//tbody//tr")
        for i, row in enumerate(table_rows):
            qtr_scores_array.append(driver.find_element_by_xpath("//td[@class='score quarter qtr" + str(i+1) + "']"))
        print(game_array)

def team_scrape():
    pass

def player_scrape():
    pass
