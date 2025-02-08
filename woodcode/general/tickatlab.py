from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from getpass import getpass, GetPassWarning
import time


def get_tickatlab_info(animal_id, username=None, password=None):
    url = "https://www.resource-manager.brr.mvm.ed.ac.uk/tickatlab/default.aspx"
    driver = None

    if username is None:
        username = input("Enter your tick@lab username: ")
    if password is None:
        try:
            password = getpass("Enter your tick@lab password: ")
        except GetPassWarning:
            print("Note: Password will be visible when typing")
            password = input("Enter your tick@lab password: ")

    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 20)

        print("Starting login process...")
        driver.get(url)
        username_field = wait.until(EC.presence_of_element_located((By.ID, "login")))
        username_field.send_keys(username)
        submit_button = wait.until(EC.element_to_be_clickable((By.ID, "submit")))
        submit_button.click()

        password_field = wait.until(EC.presence_of_element_located((By.ID, "password")))
        password_field.send_keys(password)
        submit_button = wait.until(EC.element_to_be_clickable((By.ID, "submit")))
        submit_button.click()
        print("Login completed")

        print("Navigating to breeding page...")
        breeding_link = wait.until(EC.element_to_be_clickable((By.ID, "db180813MWE7")))
        driver.execute_script("arguments[0].click();", breeding_link)

        display_dropdown = wait.until(EC.presence_of_element_located((By.ID, "A100503DBE01--1S1")))
        select = Select(display_dropdown)
        select.select_by_value("STOCKANDEXITED")
        time.sleep(2)

        print(f"Searching for animal ID: {animal_id}")
        id_field = wait.until(EC.presence_of_element_located((By.ID, "A100928MWE1--1S1")))
        id_field.clear()
        id_field.send_keys(str(animal_id))

        apply_button = wait.until(EC.presence_of_element_located((
            By.XPATH, "//span[@class='USEFILTER button__label']/parent::*"
        )))
        driver.execute_script("arguments[0].click();", apply_button)

        print("Waiting for results...")
        time.sleep(4)

        tables = driver.find_elements(By.TAG_NAME, "table")
        target_table = None

        print("Scanning tables for animal data...")
        for table in tables:
            cells = table.find_elements(By.TAG_NAME, "td")
            for cell in cells:
                if str(animal_id) in cell.text:
                    target_table = table
                    break
            if target_table:
                break

        if not target_table:
            print("Could not find table with animal data")
            return None

        print("Getting animal data...")
        cells = target_table.find_elements(By.TAG_NAME, "td")

        # Updated field names and added new fields
        animal_data = {
            'Animal_ID': cells[4].text.split("//")[0] if len(cells) > 4 else "N/A",
            'Strain': cells[7].text if len(cells) > 7 else "N/A",
            'Genotype': cells[8].text if len(cells) > 8 else "N/A",
            'DoB': cells[9].text if len(cells) > 9 else "N/A",
            'Age': cells[10].text if len(cells) > 10 else "N/A",
            'Exit_date': cells[11].text if len(cells) > 11 else "N/A",
            'No_of_animals': cells[12].text if len(cells) > 12 else "N/A",
            'Sire': cells[13].text if len(cells) > 13 else "N/A",
            'Dam': cells[14].text if len(cells) > 14 else "N/A",
            'Team': cells[15].text if len(cells) > 15 else "N/A",
            'PPL': cells[16].text if len(cells) > 16 else "N/A",
            'Protocol': cells[17].text if len(cells) > 17 else "N/A",
            'Room': cells[18].text if len(cells) > 18 else "N/A",
            'Status': cells[19].text if len(cells) > 19 else "N/A",
            'Project_code': cells[20].text if len(cells) > 20 else "N/A",
            'Tags': cells[21].text if len(cells) > 21 else "N/A",
            'Responsible_User': cells[22].text if len(cells) > 22 else "N/A"
        }

        print("\nExtracted Animal Data:")
        for key, value in animal_data.items():
            print(f"{key}: {value}")

        return animal_data

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        print("Full traceback:", traceback.format_exc())
        return None

    finally:
        if driver:
            driver.quit()